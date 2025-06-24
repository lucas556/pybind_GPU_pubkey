import itertools
import logging
import time
import random
import os
from multiprocessing import Pool, cpu_count, shared_memory
import pybind_derive2pub
import nayuki_crypto
import sys


class MnemonicValidator:
    wordlist = None
    index_map = None

    def __init__(self, wordlist_data=None):
        if MnemonicValidator.wordlist is None and wordlist_data:
            decoded = wordlist_data.decode("utf-8").splitlines()
            MnemonicValidator.wordlist = [line.strip() for line in decoded if line.strip()]
            MnemonicValidator.index_map = {w: i for i, w in enumerate(MnemonicValidator.wordlist)}
        self.wordlist = MnemonicValidator.wordlist
        self.index_map = MnemonicValidator.index_map

    def is_valid(self, mnemonic: str) -> bool:
        words = mnemonic.strip().split()
        if len(words) not in [12, 15, 18, 21, 24] or len(set(words)) != len(words):
            return False
        try:
            indices = [self.index_map[w] for w in words]
        except KeyError:
            return False
        bit_str = ''.join(f"{index:011b}" for index in indices)
        entropy_length = len(bit_str) - len(bit_str) // 33
        entropy_bits = bit_str[:entropy_length]
        checksum_bits = bit_str[entropy_length:]
        entropy_bytes = int(entropy_bits, 2).to_bytes(entropy_length // 8, byteorder="big")
        hash_bytes = nayuki_crypto.sha256(entropy_bytes)
        hash_bits = bin(int.from_bytes(hash_bytes, "big"))[2:].zfill(256)
        expected_checksum = hash_bits[:len(checksum_bits)]
        return checksum_bits == expected_checksum


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

custom_words = """
fire cigar ......
""".split()

path_indices = [44 | 0x80000000, 195 | 0x80000000, 0 | 0x80000000, 0, 0]
passphrase = ""
threads_per_block = 256
batch_size_gpu = 20000
batch_size_cpu = 100000

checkpoint_path = "progress_checkpoint.txt"
target_pubkey_hex = "04......"  # add "04"
target_pubkey_bytes = bytes.fromhex(target_pubkey_hex)


def validate_batch(batch):
    shm = shared_memory.SharedMemory(name="wordlist_shm")
    wordlist_data = bytes(shm.buf[:])
    validator = MnemonicValidator(wordlist_data=wordlist_data)
    return [' '.join(words) for words in batch if validator.is_valid(' '.join(words))]


def batched(it, size):
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def main():
    logger.info("Loading wordlist and preparing shared memory...")
    with open("english.txt", "rb") as f:
        wordlist_data = f.read()
    shm = shared_memory.SharedMemory(create=True, size=len(wordlist_data), name="wordlist_shm")
    shm.buf[:len(wordlist_data)] = wordlist_data
    MnemonicValidator(wordlist_data=wordlist_data)

    logger.info("Generating all 12-word combinations...")
    all_combinations = itertools.combinations(custom_words, 12)

    # 恢复进度
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            start_batch_id = int(f.read().strip())
        logger.info(f"[Resume] Skipping to batch {start_batch_id}...")
        all_combinations = itertools.islice(all_combinations, start_batch_id * batch_size_cpu, None)
    else:
        start_batch_id = 0

    logger.info("Validating mnemonics and deriving pubkeys on-the-fly...")
    pool = Pool(cpu_count() * 2)
    batch_id = start_batch_id
    total_pubkeys = 0

    with open("matched_mnemonics.txt", "w", encoding="utf-8") as fout:
        for batch in batched(all_combinations, batch_size_cpu):
            start = time.time()
            results = pool.map(validate_batch, [batch])
            valid_batch = [m for group in results for m in group]
            logger.info(f"[CPU] Batch {batch_id}: validated {len(batch)} → {len(valid_batch)} valid in {time.time() - start:.2f}s")

            if valid_batch:
                start_gpu = time.time()
                pubkeys = pybind_derive2pub.derive_pubkeys(
                    mnemonics=valid_batch,
                    passphrase=passphrase,
                    path_indices=path_indices,
                    threads_per_block=threads_per_block
                )
                logger.info(f"[GPU] Derived {len(pubkeys)} pubkeys in {time.time() - start_gpu:.2f}s")

                sample_idx = random.randint(0, len(valid_batch) - 1)
                logger.info(f"[Sample] {valid_batch[sample_idx]} → {bytes(pubkeys[sample_idx]).hex()}")

                for mnemonic, pubkey in zip(valid_batch, pubkeys):
                    if bytes(pubkey) == target_pubkey_bytes:
                        logger.info(f"[✓] MATCH FOUND! {mnemonic} → {target_pubkey_hex}")
                        fout.write(mnemonic + "\n")
                        fout.flush()
                        pool.terminate()
                        pool.join()
                        shm.close()
                        shm.unlink()
                        sys.exit(0)

                total_pubkeys += len(pubkeys)

            if batch_id % 10 == 0:
                with open(checkpoint_path, "w") as f:
                    f.write(str(batch_id))
            batch_id += 1

    pool.close()
    pool.join()
    shm.close()
    shm.unlink()
    logger.info(f"[✓] Total pubkeys derived: {total_pubkeys}")


if __name__ == "__main__":
    main()
