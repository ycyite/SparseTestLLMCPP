#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif
static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;
static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}


int main(int argc, char ** argv){
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

        // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model;
    ctx = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    std::cout << "!!!!!!!" << std::endl;
    std::cout << "model size: " << llama_model_size(model) << std::endl;
    std::cout << "the number of model parameter: " << llama_model_n_params(model) << std::endl;
    auto tensors = llama_get_all_tensors(model);
    int i = 0;
    float *data = nullptr;
    int total_elements = 1;
    const std::string output_folder = "../unsparsed_parameters";
    std::string tensor_name;
    std::string file_path;
    long long zero_num = 0;
    long long tensor_zeros = 0;
    for (auto tensor : tensors){
        tensor_name.assign(ggml_get_name(tensor));
        file_path = output_folder + "/" + tensor_name + ".txt";
        std::ofstream txt_file(file_path);
        data = (float*)tensor->data;
        total_elements = 1;
        tensor_zeros = 0;
        for(int i = 0; i < GGML_MAX_DIMS; i++){
            if(tensor->ne[i] > 0){
                total_elements *= tensor->ne[i];
            }
        }
        
        for (int i = 0; i < total_elements; i++){
            if(i < 900){
                txt_file << data[i] << std::endl;
            }
            // if(data[i] == 0){
            //     tensor_zeros++;
            // }
        }
        
    
        //txt_file << "The number of " + tensor_name + "'s zeros is " << tensor_zeros << "." << std::endl;
        txt_file.close();
        std::cout <<"The tensor " + tensor_name + "'s parameters has been written into a file."<<std::endl;
    }
    return 0;
}