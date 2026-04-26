from utils.inference import run


def run_baseline(model, tokenizer, dataset):
    _, accuracy = run(model, tokenizer, dataset)
    return accuracy


if __name__=="__main__":
    from utils.data import get_datasets
    from utils.model import get_model, get_tokenizer


    _, test_data = get_datasets()

    tokenizer = get_tokenizer()
    model = get_model()

    baseline_acc = run_baseline(model, tokenizer, test_data)
    print("Baseline accuracy:", baseline_acc)