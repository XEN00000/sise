import csv

correct = 0
total   = 0

with open("test_results.csv", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        targets = [float(x) for x in row[2].strip("[]").split()]
        outputs = [float(x) for x in row[4].strip("[]").split()]
        pred_idx = outputs.index(max(outputs))
        true_idx = targets.index(max(targets))
        if pred_idx == true_idx:
            correct += 1
        total += 1

print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")