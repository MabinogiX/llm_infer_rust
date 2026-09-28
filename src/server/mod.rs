//! Model server startup and component assembly.

mod api;
mod cli;
mod manager;
mod schemas;
mod serve;
mod streaming;

pub use api::router;
pub use cli::{USAGE, parse_args};
pub use manager::{FrontendManager, IncrementalDetokenizer, ManagerError, RequestHandle};
pub use schemas::{ChatCompletionRequest, CompletionRequest};
pub use serve::{ServeArgs, ServeComponents, ServeError, build_components, serve};
