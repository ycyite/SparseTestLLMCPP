# README for LLaMA2 7b Sparsification and Test in LLaMA.cpp

## Prerequisites

1. **Server Requirements**: Ensure you have access to a server with more than **28GB of memory**.
2. **Model**: Use the LLaMA2 7b model from Hugging Face.
   - Link: [LLaMA2 7b Chat Model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main)
3. **Sparsification Tool**: Clone the following repository:
   - Link: [WANDA Sparsification Tool](https://github.com/locuslab/wanda)

## Environment Setup

1. Clone the required repositories:
   ```bash
   git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   git clone https://github.com/locuslab/wanda
   ```
2. Navigate to each repository and install the dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
3. Recommended framework and toolkit:
   - **CUDA**: Version 12.4
   - **PyTorch**: Ensure PyTorch is compatible with your CUDA version.

## Editing `prune.py`

1. Locate the `prune.py` script in the WANDA repository.
2. Modify the `prune.py` file to include your desired pruning method. Supported methods:
   - `prune_wanda`
   - `prune_sparsegpt`
   - `prune_ablate`
3. Adjust the `get_loaders` line in the script to change the dataset.
   - Example: Replace the dataset with `wikitext2` or another dataset of your choice.

## Running the Sparsification Script

1. Use the `main.py` script to sparsify the model.
2. Example usage:
   ```bash
   python main.py \
       --model decapoda-research/llama-7b-hf \
       --prune_method wanda \
       --sparsity_ratio 0.5 \
       --sparsity_type unstructured \
       --save out/llama_7b/unstructured/wanda/
   ```

## Testing the Sparsified Model

1. Clone the `llama.cpp` repository to test the sparsified model:
   - Link: [LLaMA.cpp SparseGPT](https://github.com/ggerganov/llama.cpp)
2. Build the `llama.cpp` tool by following the instructions provided in the repository.
3. Use the built tool to load and test the sparsified model.

## Converting the Sparsified Model

1. Use the `convert_hf_to_gguf.py` script to convert the sparsified model to the GGUF format.
2. Example usage:
   ```bash
   python convert_hf_to_gguf.py out/llama_7b/unstructured/wanda/ --outfile out/llama_7b/gguf/model.gguf
   ```

## Testing the GGUF Model

1. Navigate to the `build/bin` directory in the `llama.cpp` repository.
2. Use the `llama-cli` command-line tool to test the GGUF model.
3. Example usage:
   ```bash
   ./llama-cli -m out/llama_7b/gguf/model.gguf -p "I believe the meaning of life is" -n 128
   ```

## Extracting Parameters from GGUF Model

1. Navigate to the `include/llama.h` file in the `llama.cpp` repository.
2. Add the following line to declare a new API for extracting all tensors:
   ```cpp
   LLAMA_API std::vector<struct ggml_tensor*> llama_get_all_tensors(struct llama_model *model);
   ```
3. Navigate to the `src/llama.cpp` file in the `llama.cpp` repository.
4. Add the following implementation to define the new API:
   ```cpp
   std::vector<struct ggml_tensor*> llama_get_all_tensors(struct llama_model *model) {
       std::vector<struct ggml_tensor*> result;
       for (auto name : model->tensors_by_name) {
           result.push_back(llama_get_model_tensor(model, name.first.c_str()));
       }
       return result;
   }
   ```
5. Rebuild the `llama.cpp` tool to include this new functionality.
6. Navigate to the `ggml`, `src`, and `common` directories in the `llama.cpp` repository and build the components to ensure all necessary changes are integrated.

## Adding a Visualization Example

1. Navigate to the existing `examples` directory in the `llama.cpp` repository.
2. Create a new subdirectory for the visualization project, such as `visualization`.
3. Inside the `examples/visualization` directory, add a file named `visualParameters.cpp`.
4. Add the necessary implementation for parameter visualization in `visualParameters.cpp`.
5. In the same `examples/visualization` directory, create a `CMakeLists.txt` file for building the example.
6. Use the `CMakeLists.txt` file from the repository as a template and modify it to include `visualParameters.cpp`.
7. Update the `CMakeLists.txt` file to reflect the correct paths for `visualParameters.cpp` and other required files.
8. Build the `examples/visualization` directory using the following commands:
   ```bash
   cd examples/visualization
   cmake .
   make
   ```
9. Test the visualization tool using the following command:
   ```bash
   ./llama-visual --model ../../../../gguf_models/unsparsed_llama.gguf
   ```

## Executing Additional Code in Jupyter Notebook

1. Open the `main.ipynb` file in a Jupyter Notebook environment.
2. Execute the cells sequentially to perform additional validations or analyses related to sparsification.
3. Ensure all required libraries are installed in the notebook's environment before running the code.


