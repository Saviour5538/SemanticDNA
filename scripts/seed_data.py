"""Ingest 6 sample documents via the running API to test all genome dimensions."""
import sys
import urllib.request
import urllib.error
import json

BASE_URL = "http://localhost:8000"

DOCUMENTS = [
    {
        "title": "TensorFlow XLA Vulnerability",
        "content": (
            "Polymorphic deserialization vulnerability in the TensorFlow XLA compiler allows "
            "remote code execution via crafted MLIR bytecode. The CVE-2024-3841 patch introduced "
            "a regression in int8 quantization pipelines affecting distributed training workloads. "
            "Mitigation requires upgrading libtensorflow to version 2.14.1 or applying the "
            "backported hotfix for CUDA kernels."
        ),
    },
    {
        "title": "The Quick Brown Fox",
        "content": (
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter "
            "of the English alphabet and is commonly used in typography tests. The dog was very "
            "lazy and did not move. It just sat there and watched the fox run by."
        ),
    },
    {
        "title": "Pharmacokinetics Overview",
        "content": (
            "Pharmacokinetics of the drug demonstrates biphasic elimination with a half-life "
            "of four hours. Cytochrome P450 enzymes metabolize the compound through hydroxylation "
            "pathways. Bioavailability is reduced by first-pass metabolism in the hepatic portal "
            "circulation. Therapeutic drug monitoring is recommended for patients with renal "
            "impairment or concomitant CYP3A4 inhibitors."
        ),
    },
    {
        "title": "Pharmakokenetics (Misspelled Version)",
        "content": (
            "Farmakokenetics of the drg demonstrate byefasic eliminashun with a half-lyfe of "
            "four ours. Sytochrome P450 ensymes metabolyze the compund through hydroksylashun "
            "pathwayz. Byoavailability is redused by first-pass metabulism in the hepatik "
            "portel sirkulation."
        ),
    },
    {
        "title": "Neural Networks and GPU Clusters",
        "content": (
            "Large neural networks require substantial computational resources for training "
            "distributed systems. Modern GPU clusters enable efficient parallel matrix operations "
            "across thousands of cores. Gradient descent optimization converges faster with "
            "mixed-precision arithmetic and adaptive learning rate schedulers. Transformer "
            "architectures leverage self-attention mechanisms for sequence modelling."
        ),
    },
    {
        "title": "Commercial Aviation Maintenance",
        "content": (
            "Large commercial aircraft require substantial maintenance schedules for operating "
            "international routes reliably. Modern ground crews enable efficient parallel runway "
            "operations across multiple terminals. Safety inspection protocols converge faster "
            "with automated diagnostic systems and adaptive scheduling software. Wide-body "
            "airframes leverage composite materials for structural load distribution."
        ),
    },
]


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    print(f"Seeding {len(DOCUMENTS)} documents into {BASE_URL}...\n")
    for i, doc in enumerate(DOCUMENTS, 1):
        try:
            result = post("/documents", doc)
            print(f"[{i}/{len(DOCUMENTS)}] Ingested: #{result['id']} — {doc['title']}")
            print(f"           Genome: {result['genome_length']} genes extracted")
        except urllib.error.URLError as e:
            print(f"ERROR: Could not reach API at {BASE_URL}. Is the server running?")
            print(f"       {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR on doc {i}: {e}")

    print("\nSeed complete. Try these queries:")
    print('  curl -s -X POST http://localhost:8000/search \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"query": "farmakokenetics drug elimination"}\' | python -m json.tool')
    print()
    print('  curl -s -X POST http://localhost:8000/search \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"query": "neural networks need computing power"}\' | python -m json.tool')


if __name__ == "__main__":
    main()
