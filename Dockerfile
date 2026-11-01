# Mount an index at /index. See docs/deploying.md.
FROM rust:1-slim AS build
WORKDIR /src
COPY . .
RUN cargo build --release -p unsaid

FROM debian:stable-slim
COPY --from=build /src/target/release/unsaid /usr/local/bin/unsaid
EXPOSE 8377
CMD ["unsaid", "serve", "/index", "--host", "0.0.0.0"]
