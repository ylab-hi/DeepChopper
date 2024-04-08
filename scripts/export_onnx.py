import torch
from deepchopper.models.cnn import BenchmarkCNN

def cnn():
    vocab_size = 11  # number of unique tokens
    num_filters = [128, 256, 512]  # number of filters in each convolutional layer
    filter_sizes = [7,9,11]   # filter sizes in each convolutional layer
    return BenchmarkCNN(2, vocab_size, num_filters, filter_sizes, 100)

def main():
    # Parameters for the model
    test_input = torch.load("./tests/data/input_ids.pt") # [batch_size, input_len]
    test_quals = torch.load("./tests/data/input_quals.pt") # [batch_size, input_len]
    cnn_model = cnn()

    # Assume cnn_model is your model and test_input is the input tensor
    onnx_export_output = torch.onnx.export(cnn_model, test_input, test_quals, "model.onnx")


if __name__ == "__main__":
    main()
