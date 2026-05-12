{
  description = "droplesim — waLBerla + lbmpy microfluidics simulation environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;  # CUDA
        };
        python = pkgs.python311;
      in {
        devShells.default = pkgs.mkShell {
          name = "droplesim";

          packages = with pkgs; [
            # Build toolchain
            cmake
            ninja
            gcc13
            pkg-config

            # CUDA (override with local install path on remote server)
            # cudaPackages.cudatoolkit
            # cudaPackages.cuda_nvcc

            # Python
            python
            uv

            # System libs needed by lbmpy/pystencils
            stdenv.cc.cc.lib
            zlib

            # HDF5 (C library, h5py links against it)
            hdf5

            # Utilities
            git
            jq
          ];

          shellHook = ''
            echo "droplesim dev shell"
            echo "  uv sync       — install Python deps"
            echo "  make configure — CMake configure (downloads waLBerla)"
            echo "  make build    — compile apps"
            echo ""
            # Point uv/pip at the system HDF5 so h5py builds correctly
            export HDF5_DIR=${pkgs.hdf5}
          '';
        };
      });
}
