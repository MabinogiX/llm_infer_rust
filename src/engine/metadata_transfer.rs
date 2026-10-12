//! Scheduler-buffer uploads, separated from graph execution and KV retirement.

use std::{marker::PhantomData, rc::Rc};
use tch::{Device, Tensor};

use super::{CompletionFence, ModelRunnerError};
type Result<T> = std::result::Result<T, ModelRunnerError>;

#[derive(Clone)]
pub(crate) struct InputReadFence {
    #[cfg(has_cuda_graph)]
    native: Rc<NativeInputEvent>,
    device: i32,
    _same_thread: PhantomData<Rc<()>>,
}

impl InputReadFence {
    pub(crate) fn record(device: Device) -> Result<Self> {
        let Device::Cuda(device) = device else {
            return Err(unavailable());
        };
        #[cfg(has_cuda_graph)]
        {
            let ptr = unsafe { sglang_input_read_record(device as i32) };
            if ptr.is_null() {
                return Err(native_error());
            }
            Ok(Self {
                native: Rc::new(NativeInputEvent(ptr)),
                device: device as i32,
                _same_thread: PhantomData,
            })
        }
        #[cfg(not(has_cuda_graph))]
        {
            let _ = device;
            Err(unavailable())
        }
    }
    fn wait_current(&self) -> Result<()> {
        #[cfg(has_cuda_graph)]
        if !unsafe { sglang_input_read_wait(self.native.0, self.device) } {
            return Err(native_error());
        }
        #[cfg(not(has_cuda_graph))]
        let _ = self.device;
        Ok(())
    }
}

#[cfg(has_cuda_graph)]
struct NativeInputEvent(*mut std::ffi::c_void);
#[cfg(has_cuda_graph)]
impl Drop for NativeInputEvent {
    fn drop(&mut self) {
        unsafe { sglang_input_read_free(self.0) };
    }
}

pub(crate) struct MetadataUpload {
    #[cfg(has_cuda_graph)]
    ptr: *mut std::ffi::c_void,
    _same_thread: PhantomData<Rc<()>>,
}
pub(crate) struct UploadGuard<'a> {
    upload: &'a MetadataUpload,
    ended: bool,
}

impl MetadataUpload {
    pub(crate) fn new(device: Device) -> Result<Self> {
        let Device::Cuda(device) = device else {
            return Err(unavailable());
        };
        #[cfg(has_cuda_graph)]
        {
            let ptr = unsafe { sglang_metadata_upload_create(device as i32) };
            if ptr.is_null() {
                return Err(native_error());
            }
            Ok(Self {
                ptr,
                _same_thread: PhantomData,
            })
        }
        #[cfg(not(has_cuda_graph))]
        {
            let _ = device;
            Err(unavailable())
        }
    }
    pub(crate) fn begin(
        &self,
        early: Option<&InputReadFence>,
        full: Option<&CompletionFence>,
    ) -> Result<UploadGuard<'_>> {
        #[cfg(has_cuda_graph)]
        if !unsafe { sglang_metadata_upload_begin(self.ptr) } {
            return Err(native_error());
        }
        let guard = UploadGuard {
            upload: self,
            ended: false,
        };
        if let Some(early) = early {
            early.wait_current()?;
        } else if let Some(full) = full {
            full.wait_current()
                .map_err(|error| ModelRunnerError::Model(error.to_string()))?;
        }
        Ok(guard)
    }
}
impl UploadGuard<'_> {
    pub(crate) fn copy(&self, target: &mut Tensor, source: &Tensor) -> Result<()> {
        #[cfg(has_cuda_graph)]
        if !unsafe {
            sglang_metadata_upload_copy(
                self.upload.ptr,
                target.as_mut_ptr().cast(),
                source.as_ptr().cast(),
            )
        } {
            return Err(native_error());
        }
        #[cfg(not(has_cuda_graph))]
        {
            let _ = (target, source);
            return Err(unavailable());
        }
        #[cfg(has_cuda_graph)]
        Ok(())
    }
    pub(crate) fn finish(mut self) -> Result<()> {
        #[cfg(has_cuda_graph)]
        let ok = unsafe { sglang_metadata_upload_end(self.upload.ptr) };
        self.ended = true;
        #[cfg(has_cuda_graph)]
        if !ok {
            return Err(native_error());
        }
        Ok(())
    }
}
impl Drop for UploadGuard<'_> {
    fn drop(&mut self) {
        #[cfg(has_cuda_graph)]
        if !self.ended {
            unsafe { sglang_metadata_upload_end(self.upload.ptr) };
        }
        #[cfg(not(has_cuda_graph))]
        let _ = self.upload;
    }
}
impl Drop for MetadataUpload {
    fn drop(&mut self) {
        #[cfg(has_cuda_graph)]
        unsafe {
            sglang_metadata_upload_free(self.ptr)
        };
    }
}
fn unavailable() -> ModelRunnerError {
    ModelRunnerError::Model("CUDA metadata upload unavailable".into())
}
#[cfg(has_cuda_graph)]
fn native_error() -> ModelRunnerError {
    ModelRunnerError::Model(
        unsafe { std::ffi::CStr::from_ptr(sglang_graph_error()) }
            .to_string_lossy()
            .into_owned(),
    )
}
#[cfg(has_cuda_graph)]
unsafe extern "C" {
    fn sglang_graph_error() -> *const std::ffi::c_char;
    fn sglang_input_read_record(device: i32) -> *mut std::ffi::c_void;
    fn sglang_input_read_wait(event: *mut std::ffi::c_void, device: i32) -> bool;
    fn sglang_input_read_free(event: *mut std::ffi::c_void);
    fn sglang_metadata_upload_create(device: i32) -> *mut std::ffi::c_void;
    fn sglang_metadata_upload_begin(upload: *mut std::ffi::c_void) -> bool;
    fn sglang_metadata_upload_copy(
        upload: *mut std::ffi::c_void,
        target: *mut std::ffi::c_void,
        source: *const std::ffi::c_void,
    ) -> bool;
    fn sglang_metadata_upload_end(upload: *mut std::ffi::c_void) -> bool;
    fn sglang_metadata_upload_free(upload: *mut std::ffi::c_void);
}

#[cfg(all(test, has_cuda_graph))]
mod tests {
    use super::*;
    use tch::Kind;

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn uploads_preserve_host_sources_restore_stream_and_order_buffer_reuse() {
        let device = Device::Cuda(0);
        let context = crate::engine::NativeGraphCaptureContext::new(0).unwrap();
        let upload = MetadataUpload::new(device).unwrap();
        let mut target = Tensor::zeros([16], (Kind::Int64, device));
        let mut read_done = InputReadFence::record(device).unwrap();
        let mut retained = Vec::new();
        for iteration in 0..64 {
            let guard = upload.begin(Some(&read_done), None).unwrap();
            assert!(context.check_execution_stream().is_err());
            let source = Tensor::arange(16, (Kind::Int64, Device::Cpu)) + iteration * 100;
            guard.copy(&mut target, &source).unwrap();
            drop(source); // Native pinned storage must outlive this CPU tensor.
            if iteration % 2 == 0 {
                guard.finish().unwrap();
            } else {
                drop(guard);
            }
            context.check_execution_stream().unwrap();
            retained.push(target.copy());
            // The next upload may overwrite target only after this snapshot.
            read_done = InputReadFence::record(device).unwrap();
        }
        drop(upload); // Must fence host/destination use before freeing storage.
        for (iteration, output) in retained.iter().enumerate() {
            assert_eq!(
                Vec::<i64>::try_from(output).unwrap(),
                (0..16)
                    .map(|row| row + iteration as i64 * 100)
                    .collect::<Vec<_>>()
            );
        }
    }
}
