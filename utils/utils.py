import os

def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_text_file(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def split_text_into_chunks(text: str, max_len: int) -> list:
    paras = text.split("\n\n")
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_len:
            buf += p + "\n\n"
        else:
            if buf: chunks.append(buf.strip())
            if len(p) <= max_len:
                buf = p + "\n\n"
            else:
                # fallback split by sentences
                sents = p.split('. ')
                sub = ""
                for s in sents:
                    if len(sub) + len(s) + 2 <= max_len:
                        sub += s + '. '
                    else:
                        chunks.append(sub.strip())
                        sub = s + '. '
                if sub: chunks.append(sub.strip())
                buf = ""
    if buf.strip(): chunks.append(buf.strip())
    return chunks