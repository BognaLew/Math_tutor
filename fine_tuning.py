from utils.fine_tune_utils import fine_tune
from utils.inference import run

def run_fine_tuning(model, tokenizer, dataset):
    trained_model, tokenizer = fine_tune(model, tokenizer, dataset)

    _, accuracy = run(trained_model, tokenizer, dataset)
    return accuracy


if __name__=="__main__":
    from utils.data import get_datasets
    from utils.model import get_model, get_tokenizer


    _, test_data = get_datasets()

    tokenizer = get_tokenizer()
    model = get_model()

    acc = run_fine_tuning(model, tokenizer, test_data)
    print("Fine-tuning accuracy:", acc)