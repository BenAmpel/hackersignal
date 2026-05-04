from etg.rt2.trigram_hasher import PAD_ID, CLS_ID, token_hash, tokenize_cte


def test_token_hash_deterministic():
    assert token_hash("sqli", 1024) == token_hash("sqli", 1024)
    # Capitalization ignored (lowered).
    assert token_hash("SqLi", 1024) == token_hash("sqli", 1024)


def test_tokenize_cte_padding():
    ids = tokenize_cte("sqli exploit apache", n_buckets=1024, max_len=8)
    assert len(ids) == 8
    # First id is CLS.
    assert ids[0] == CLS_ID
    # Trailing padding IDs.
    assert ids[-1] in (PAD_ID, CLS_ID) or ids.count(PAD_ID) > 0


def test_tokenize_truncation():
    ids = tokenize_cte("a b c d e f g h i j", n_buckets=1024, max_len=4)
    assert len(ids) == 4
    assert ids[0] == CLS_ID
