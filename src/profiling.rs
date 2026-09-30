//! Optional request-stage profiling. CUDA is synchronized only in diagnostic mode.

use std::{future::Future, time::Instant};

use tch::{Cuda, Device};

use crate::{engine::BatchPhase, logging::step_timing_enabled, scheduler::RequestId};

#[derive(Clone, Copy)]
pub(crate) enum ModelStage {
    Plan,
    Embed,
    Layers,
    Norm,
    QkvLinear,
    QkNorm,
    Rope,
    KvWrite,
    AttentionBackend,
    AttentionOutput,
    Mlp,
    Head,
    PlanDrop,
}

const MODEL_STAGE_COUNT: usize = ModelStage::PlanDrop as usize + 1;

pub(crate) struct ModelProfiler {
    enabled: bool,
    device: Device,
    started: Option<Instant>,
    us: [u128; MODEL_STAGE_COUNT],
}

pub(crate) struct ModelStageTimer {
    stage: ModelStage,
    started: Option<Instant>,
}

impl ModelProfiler {
    pub(crate) fn new(device: Device) -> Self {
        let enabled = step_timing_enabled();
        Self {
            enabled,
            device,
            started: enabled.then(Instant::now),
            us: [0; MODEL_STAGE_COUNT],
        }
    }

    /// The timer owns its start time and does not borrow the profiler, so a
    /// layer can record nested stages without changing the model's expressions.
    pub(crate) fn start(&self, stage: ModelStage) -> ModelStageTimer {
        ModelStageTimer {
            stage,
            started: self.enabled.then(Instant::now),
        }
    }

    pub(crate) fn finish(&mut self, timer: ModelStageTimer) {
        if let Some(started) = timer.started {
            synchronize(self.device);
            self.us[timer.stage as usize] += started.elapsed().as_micros();
        }
    }

    pub(crate) fn log(&self, phase: BatchPhase, tokens: usize, layers: usize) {
        let Some(started) = self.started else { return };
        let us = |stage: ModelStage| self.us[stage as usize];
        tracing::info!(
            ?phase,
            tokens,
            layers,
            plan_us = us(ModelStage::Plan),
            embed_us = us(ModelStage::Embed),
            layers_us = us(ModelStage::Layers),
            norm_us = us(ModelStage::Norm),
            qkv_rope_us = us(ModelStage::QkvLinear) + us(ModelStage::QkNorm) + us(ModelStage::Rope),
            qkv_linear_us = us(ModelStage::QkvLinear),
            qk_norm_us = us(ModelStage::QkNorm),
            rope_us = us(ModelStage::Rope),
            kv_write_us = us(ModelStage::KvWrite),
            attention_backend_us = us(ModelStage::AttentionBackend),
            attention_output_us = us(ModelStage::AttentionOutput),
            mlp_us = us(ModelStage::Mlp),
            head_us = us(ModelStage::Head),
            plan_drop_us = us(ModelStage::PlanDrop),
            model_total_us = started.elapsed().as_micros(),
            "model forward timing"
        );
    }
}

#[derive(Clone, Copy)]
pub(crate) enum StepStage {
    Params,
    Forward,
    Sample,
    Publish,
}

pub(crate) struct StepProfiler {
    enabled: bool,
    device: Device,
    started: Option<Instant>,
    schedule_us: u128,
    us: [u128; 4],
}

impl StepProfiler {
    pub(crate) fn new(device: Device, schedule_us: u128) -> Self {
        let enabled = step_timing_enabled();
        Self {
            enabled,
            device,
            started: enabled.then(Instant::now),
            schedule_us,
            us: [0; 4],
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn measure<T>(&mut self, stage: StepStage, work: impl FnOnce() -> T) -> T {
        let Some(started) = self.enabled.then(Instant::now) else {
            return work();
        };
        let value = work();
        synchronize(self.device);
        self.us[stage as usize] += started.elapsed().as_micros();
        value
    }

    pub(crate) fn log(&self, phase: BatchPhase, requests: &[(RequestId, usize)], success: bool) {
        let Some(started) = self.started else { return };
        let [params_us, forward_us, sample_us, publish_us] = self.us;
        let schedule_us = self.schedule_us;
        let phase_total_us = schedule_us + started.elapsed().as_micros();
        for &(uid, token_index) in requests {
            tracing::info!(
                uid,
                token_index,
                ?phase,
                batch_size = requests.len(),
                schedule_us,
                params_us,
                forward_us,
                sample_us,
                publish_us,
                phase_total_us,
                success,
                "request step timing"
            );
        }
    }
}

pub(crate) fn time_schedule<T>(device: Device, work: impl FnOnce() -> T) -> (T, u128) {
    let Some(started) = step_timing_enabled().then(Instant::now) else {
        return (work(), 0);
    };
    let value = work();
    synchronize(device);
    (value, started.elapsed().as_micros())
}

fn synchronize(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RequestStage {
    Template,
    Encode,
    Submit,
    Wait,
    Format,
}

pub(crate) struct RequestProfiler {
    enabled: bool,
    us: [u128; 5],
}

impl RequestProfiler {
    pub(crate) fn new() -> Self {
        Self {
            enabled: step_timing_enabled(),
            us: [0; 5],
        }
    }

    pub(crate) fn measure<T>(&mut self, stage: RequestStage, work: impl FnOnce() -> T) -> T {
        let Some(started) = self.enabled.then(Instant::now) else {
            return work();
        };
        let value = work();
        self.us[stage as usize] += started.elapsed().as_micros();
        value
    }

    pub(crate) async fn measure_async<T, F: Future<Output = T>>(
        &mut self,
        stage: RequestStage,
        work: F,
    ) -> T {
        let Some(started) = self.enabled.then(Instant::now) else {
            return work.await;
        };
        let value = work.await;
        self.us[stage as usize] += started.elapsed().as_micros();
        value
    }

    pub(crate) fn log_ingress(&self, uid: RequestId, endpoint: &'static str) {
        if !self.enabled {
            return;
        }
        let encode_us = self.us[RequestStage::Encode as usize];
        let submit_us = self.us[RequestStage::Submit as usize];
        if endpoint == "chat.completions" {
            let template_us = self.us[RequestStage::Template as usize];
            tracing::info!(
                uid,
                endpoint,
                template_us,
                encode_us,
                submit_us,
                "request ingress timing"
            );
        } else {
            tracing::info!(
                uid,
                endpoint,
                encode_us,
                submit_us,
                "request ingress timing"
            );
        }
    }

    pub(crate) fn log_output(&self, uid: RequestId) {
        if !self.enabled {
            return;
        }
        let wait_us = self.us[RequestStage::Wait as usize];
        let format_us = self.us[RequestStage::Format as usize];
        tracing::info!(uid, wait_us, format_us, "request output timing");
    }
}
