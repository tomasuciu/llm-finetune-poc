import sys

def train_cli():
    from llm_finetune_poc.train import main
    sys.exit(main())


if __name__ == "__main__":
    train_cli()
