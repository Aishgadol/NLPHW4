import json

def load_jsonl(filename):
    """Load a JSONL file and return a list of parsed JSON objects."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def compare_jsonl(file1, file2):
    """
    Compare two JSONL files line by line.
    Returns a list of strings describing the differences.
    """
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    differences = []
    max_len = max(len(data1), len(data2))

    for i in range(max_len):
        if i >= len(data1):
            # file1 has fewer lines than file2
            differences.append(f"Line {i}: File1 is MISSING | File2 is {data2[i]}")
        elif i >= len(data2):
            # file2 has fewer lines than file1
            differences.append(f"Line {i}: File1 is {data1[i]} | File2 is MISSING")
        else:
            # Both lines exist, compare JSON objects
            if data1[i] != data2[i]:
                differences.append(f"Line {i}: \n  File1: {data1[i]}\n  File2: {data2[i]}")

    return differences

def main():
    file1 = "result.jsonl"
    file2 = "knesset_corpus.jsonl"

    diffs = compare_jsonl(file1, file2)

    # Print all differences
    print("Differences found:\n")
    for diff in diffs:
        print(diff)
        print("-" * 50)

    # Print total count of differences
    print(f"\nTotal number of differences: {len(diffs)}")

if __name__ == "__main__":
    main()
