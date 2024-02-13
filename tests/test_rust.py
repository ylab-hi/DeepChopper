from panda import sum_as_string

def test_sum_as_string_add_2():
    """Test sum_as_string function."""
    result = sum_as_string(1, 2)
    assert result == "3"
