from tqdm import tqdm

from utils import extract_answer, generate_answer


def run_baseline(model, tokenizer, dataset):
    results = []
    correct = 0

    for sample in tqdm(test_data):
        pred = generate_answer(sample["question"], model, tokenizer)
        pred_ans = extract_answer(pred)
        true_ans = extract_answer(sample["answer"])

        if pred_ans == true_ans:
            correct += 1
        results.append(pred)
    baseline_acc = correct / len(test_data)
    return results, baseline_acc


if __name__=="__main__":
    from data import get_datasets
    from model import get_model, get_tokenizer


    _, test_data = get_datasets()

    tokenizer = get_tokenizer()
    model = get_model()

    results, baseline_acc = run_baseline(model, tokenizer, test_data)
    print("Baseline accuracy:", baseline_acc)