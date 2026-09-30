//! Application-wide tracing setup with daily file rotation.

use std::{error::Error, fmt, fs, path::PathBuf, sync::OnceLock, time::Duration};

use tracing::{Event, Subscriber};
use tracing_appender::{
    non_blocking::{NonBlockingBuilder, WorkerGuard},
    rolling::{RollingFileAppender, Rotation},
};
use tracing_subscriber::{
    EnvFilter,
    fmt::{
        FmtContext,
        format::{FormatEvent, FormatFields, Writer},
        time::{FormatTime, SystemTime},
    },
    prelude::*,
    registry::LookupSpan,
};

pub const SERVICE_NAME: &str = "llm-infer-rust";

/// Opt-in profiling synchronizes CUDA between stages, so it changes throughput.
pub(crate) fn step_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SGLANG_PROFILE_STEPS").as_deref() == Ok("1"))
}

pub(crate) fn format_duration(duration: Duration) -> String {
    format!("{:.3} s", duration.as_secs_f64())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggingConfig {
    pub directory: PathBuf,
    pub level: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("logs"),
            level: None,
        }
    }
}

/// Installs one subscriber for all threads. Keep the returned guard alive
/// until shutdown so the non-blocking writer flushes buffered file events.
pub fn init_logging(config: &LoggingConfig) -> Result<WorkerGuard, Box<dyn Error + Send + Sync>> {
    fs::create_dir_all(&config.directory)?;
    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(SERVICE_NAME)
        .filename_suffix("log")
        .build(&config.directory)?;
    let (file_writer, guard) = NonBlockingBuilder::default().lossy(false).finish(appender);
    let directive = config
        .level
        .clone()
        .or_else(|| std::env::var("RUST_LOG").ok())
        .unwrap_or_else(|| "info".to_owned());
    let filter = EnvFilter::try_new(directive)?;
    let file_layer = tracing_subscriber::fmt::layer()
        .event_format(ServiceFormatter)
        .with_ansi(false)
        .with_writer(file_writer);
    let terminal_layer = tracing_subscriber::fmt::layer()
        .event_format(ServiceFormatter)
        .with_ansi(true)
        .with_writer(std::io::stdout);
    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(terminal_layer)
        .try_init()?;
    Ok(guard)
}

#[derive(Clone, Copy)]
struct ServiceFormatter;

impl<S, N> FormatEvent<S, N> for ServiceFormatter
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        SystemTime.format_time(&mut writer)?;
        write!(
            writer,
            " | {} | {} | ",
            SERVICE_NAME,
            event.metadata().level()
        )?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime as StdSystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn formats_duration_in_seconds_with_millisecond_precision() {
        assert_eq!(format_duration(Duration::from_millis(157)), "0.157 s");
        assert_eq!(format_duration(Duration::from_millis(1234)), "1.234 s");
    }

    #[test]
    fn writes_plain_file_and_ansi_terminal_logs() {
        let nonce = StdSystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("sglang-rust-log-test-{nonce}"));
        fs::create_dir_all(&directory).unwrap();
        let file_appender = RollingFileAppender::builder()
            .rotation(Rotation::NEVER)
            .filename_prefix("file")
            .filename_suffix("log")
            .build(&directory)
            .unwrap();
        let terminal_appender = RollingFileAppender::builder()
            .rotation(Rotation::NEVER)
            .filename_prefix("terminal")
            .filename_suffix("log")
            .build(&directory)
            .unwrap();
        let (writer, guard) = NonBlockingBuilder::default()
            .lossy(false)
            .finish(file_appender);
        let file_layer = tracing_subscriber::fmt::layer()
            .event_format(ServiceFormatter)
            .with_ansi(false)
            .with_writer(writer);
        let terminal_layer = tracing_subscriber::fmt::layer()
            .event_format(ServiceFormatter)
            .with_ansi(true)
            .with_writer(terminal_appender);
        let subscriber = tracing_subscriber::registry()
            .with(file_layer)
            .with(terminal_layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(request_id = 7, duration_ms = %format_duration(Duration::from_millis(157)), "request accepted");
        });
        drop(guard);

        let contents = fs::read_to_string(directory.join("file.log")).unwrap();
        let line = contents.lines().next().unwrap();
        assert!(line.split_whitespace().next().unwrap().contains('T'));
        assert!(line.contains(" | llm-infer-rust | INFO | request accepted request_id=7"));
        assert!(!line.contains("service="));
        assert!(!line.contains("level="));
        assert!(line.contains("request accepted"));
        assert!(line.contains("request_id=7"));
        assert!(line.contains("duration_ms=0.157 s"));
        assert!(!line.contains('\u{1b}'));

        let terminal = fs::read_to_string(directory.join("terminal.log")).unwrap();
        assert!(terminal.contains("request accepted"));
        assert!(terminal.contains('\u{1b}'));
        fs::remove_dir_all(directory).unwrap();
    }
}
