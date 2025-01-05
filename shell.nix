let
  # Pin to a specific nixpkgs commit for reproducibility.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/c807ed4f48d7055cad44555ced741d1651087156.tar.gz") {config.allowUnfree = true;};
  python = pkgs.python311;
  pythonPackages = pkgs.python311Packages;

  docstring-parser = python.pkgs.buildPythonPackage rec {
    pname = "docstring_parser";
    version = "0.16";
    format = "pyproject";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "U4vqvQrx4tsBRra9PKpSbDWjTWGvn9KIfzqKJ6c5qm4=";
    };

    propagatedBuildInputs = [
      pythonPackages.poetry-core
    ];
  };

  bitsandbytes = python.pkgs.buildPythonPackage rec {
    pname = "bitsandbytes";
    version = "0.45.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/99/9a/f41d252bf8b0bc5969b4dce1274cd04b7ddc541de1060dd27eca680bc1b2/bitsandbytes-0.45.0-py3-none-manylinux_2_24_x86_64.whl";
      sha256 = "0f0323de1ff1fdf8383e79bdad1283516a4c05a6fd2b44a363bf4e059422305b";
    };
  };

  galore-torch = python.pkgs.buildPythonPackage rec {
    pname = "galore-torch";
    version = "1.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/2b/b9/e9e88f989c62edefaa45df07198ce280ac0d373f5e2842686c6ece2ddb1e/galore_torch-1.0-py3-none-any.whl";
      sha256 = "7339bd6f6ea4557c5c9ae58026d67414bba2a0c7a7e7f1d69a05514ff565dd20";
    };

    propagatedBuildInputs = [
      pythonPackages.setuptools
      bitsandbytes
    ];
  };

  arguably = python.pkgs.buildPythonPackage rec {
    pname = "arguably";
    version = "1.3.0";
    format = "pyproject";
    doCheck = false;
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "9261e49d0281600e9eac3fb2e31d2022dc0d002b6370461d787b20690eb2a98d";
    };

    propagatedBuildInputs = [
      pythonPackages.poetry-core
      docstring-parser
    ];
  };

  peft = python.pkgs.buildPythonPackage rec {
    pname = "peft";
    version = "0.14.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/88/05/e58e3aaa36544d30a917814e336fc65a746f708e5874945e92999bc22fa3/peft-0.14.0-py3-none-any.whl";
      sha256 = "2f04f3a870c3baf30f15e7dcaa5dd70d3e54cfdd146d3c6c187735d3ae0a0700";
    };
  };

  huggingface-hub = python.pkgs.buildPythonPackage rec {
    pname = "huggingface-hub[cli]";
    version = "0.25.2";
    format = "pyproject";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "a1014ea111a5f40ccd23f7f7ba8ac46e20fa3b658ced1f86a00c75c06ec6423c";
    };

    propagatedBuildInputs = [
      pythonPackages.poetry-core
    ];
  };

  python-with-packages = python.withPackages (ps: with ps;
    let
      accelerate-bin = accelerate.override { torch = torch-bin; };
      lightning-bin = pytorch-lightning.override { torch = torch-bin; };
    in [
      torch-bin
      torchvision-bin
      galore-torch
      datasets
      transformers
      peft
      evaluate
      accelerate-bin
      pip
      lightning-bin
      arguably
      loguru
      tensorboard
      tensorboardx
      wandb
    ]);
in pkgs.mkShell {
  packages = [
    python-with-packages
  ];

  shellHook = ''
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export CUDA_PATH=${pkgs.cudatoolkit}

    source .env
    huggingface-cli login --token $HF_TOKEN
  '';
}