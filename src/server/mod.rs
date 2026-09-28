//! Model server startup and component assembly.

mod api;
mod cli;
mod components;
mod manager;
pub mod output;
mod schemas;
mod serve;
mod streaming;

pub use api::router;
pub use cli::{USAGE, parse_args};
pub use components::{ServeComponents, build_components};
pub use manager::{FrontendManager, IncrementalDetokenizer, ManagerError, RequestHandle};
pub use schemas::{ChatCompletionRequest, CompletionRequest};
pub use serve::{ServeArgs, ServeError, serve};
