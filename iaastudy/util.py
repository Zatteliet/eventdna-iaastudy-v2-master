import random
import string


def recursive_delete(dirpath):
    # deleted everything in a dir, recursively.
    for item in dirpath.iterdir():
        if item.is_dir():
            recursive_delete(item)
            item.rmdir()
        else:
            item.unlink()


def four_char_code():
    candidates = string.ascii_lowercase + string.digits + string.digits
    code = [random.choice(candidates) for _ in range(4)]
    return "".join(code)
