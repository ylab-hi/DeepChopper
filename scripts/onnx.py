import torch 
from deepchopper.models.cnn import BenchmarkCNN

def cnn():
    vocab_size = 11  # number of unique tokens
    num_filters = [128, 256, 512]  # number of filters in each convolutional layer
    filter_sizes = [7,9,11]   # filter sizes in each convolutional layer
    return BenchmarkCNN(2, vocab_size, num_filters, filter_sizes, 100)

def main():
    # Parameters for the model
    test_input = torch.randn(1, 1000, 100) # [batch_size, input_len, embedding_dim]
    test_quals = torch.randn(1, 1000) # [batch_size, input_len]
    cnn_model = cnn()
    onnx_program = torch.onnx.dynamo_export(cnn_model, test_input, test_quals)


if __name__ == "__main__":
    main()
