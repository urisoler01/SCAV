from Lab1.dwt import DWT


def test_dwt_instance_method():
    input_data = [1, 2, 3, 4, 5]
    encoded_data = DWT.encode(input_data)
    decoded_data = DWT.decode(encoded_data)
    
    assert decoded_data == input_data, "The decoded data does not match the input data."
