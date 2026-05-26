# infra/nix/flake.nix
{
  description = "STRATUM development environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable."1.76.0".default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            go_1_22
            python311
            buf          # Proto linting and codegen
            protobuf
            grpcurl
            docker-compose
            terraform
            jq
            yq
            # Observability tooling
            prometheus
            grafana
          ];
          shellHook = ''
            echo "STRATUM dev environment loaded"
            echo "Rust: $(rustc --version)"
            echo "Go: $(go version)"
            echo "Python: $(python --version)"
          '';
        };
      });
}