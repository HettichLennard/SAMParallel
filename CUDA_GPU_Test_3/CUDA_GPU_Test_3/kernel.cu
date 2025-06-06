#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <thread>

#include "System.h"
#include "ParseUtils.h"
#include "Options.h"
#include "SimpSolver.h"

using namespace Minisat;
using namespace std;

__global__
void init_rand(curandState* state, int a, int b, int feature_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < feature_count) {
        curand_init(a + index, b + 2 * index, 0, &state[index]);
    }
}


__global__
void calculate_clause_values(int clause_count, short* current_clause_values, short* min_clause_values, int* clause_list, int lits_in_clause_limit, short* clause_list_sizes, char* optimized_variant) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < clause_count) {
        min_clause_values[index] = 1;
        current_clause_values[index] = 0;

        int lit_value;

        for (int i = 0; i < clause_list_sizes[index]; i++) {
            lit_value = clause_list[lits_in_clause_limit * index + i];
            if (lit_value < 0) {
                min_clause_values[index]--;
                lit_value = -lit_value - 1;
                current_clause_values[index] -= optimized_variant[lit_value];
            }
            else {
                lit_value = lit_value - 1;
                current_clause_values[index] += optimized_variant[lit_value];
            }
        }
    }
}


__global__
void reset_uncovered(int feature_count, int* uncovered) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count && index > 0) {
        uncovered[index - 1] = 0;
    }
}


__global__
void calculate_interactions(int parallel_limit, int feature_count, unsigned long long feature_interaction_count, int* current_sample_size, char* current_sample, int* skip_sample_index, char* feature_interactions, int* uncovered) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;

        unsigned long long inter_index;

        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;
        char value_i, value_j;

        int lit_i_index, lit_j_index;

        double temp_i_value;

        while (loop_index * parallel_limit + index < feature_interaction_count) {

            inter_index = loop_index * parallel_limit + index;

            temp_i_value = 0.5 * (sqrt(double(inter_index) * 8.0 + 1.0) + 1.0);
            temp_i_value = floor(temp_i_value);

            lit_i_index = int(temp_i_value);

            lit_j_index = inter_index - ((lit_i_index - 1) * lit_i_index) / 2;

            inter_off_off = 1;
            inter_off_on = 1;
            inter_on_off = 1;
            inter_on_on = 1;

            for (int sample = 0; sample < current_sample_size[0]; sample++) {
                if (skip_sample_index[0] != sample) {
                    value_i = current_sample[lit_i_index + feature_count * sample];
                    value_j = current_sample[lit_j_index + feature_count * sample];

                    if (value_i == 0 && value_j == 0) inter_off_off = 0;
                    else if (value_i == 0 && value_j == 1) inter_off_on = 0;
                    else if (value_i == 1 && value_j == 0) inter_on_off = 0;
                    else if (value_i == 1 && value_j == 1) inter_on_on = 0;

                    if (inter_off_off == 0 && inter_off_on == 0 && inter_on_off == 0 && inter_on_on == 0) break;
                }
            }

            if (inter_off_off == 0 && inter_off_on == 0 && inter_on_off == 0 && inter_on_on == 0) {
                feature_interactions[inter_index] = 0;
            }
            else {
                inter_bin_encoding = inter_off_off + 2 * inter_off_on + 4 * inter_on_off + 8 * inter_on_on;

                feature_interactions[inter_index] = inter_bin_encoding;

                atomicAdd(&uncovered[lit_i_index - 1], inter_off_off + inter_off_on + inter_on_off + inter_on_on);
            }

            loop_index++;
        }
    }
}


__global__
void append_interactions(int parallel_limit, int feature_count, unsigned long long feature_interaction_count, char* optimized_variant, char* feature_interactions, int* uncovered) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;

        unsigned long long inter_index;

        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;
        char value_i, value_j;

        int lit_i_index, lit_j_index;

        double temp_i_value;

        while (loop_index * parallel_limit + index < feature_interaction_count) {

            inter_index = loop_index * parallel_limit + index;

            inter_bin_encoding = feature_interactions[inter_index];

            if (inter_bin_encoding != 0) {

                if (inter_bin_encoding >= 8) {
                    inter_on_on = 1;
                    inter_bin_encoding -= 8;
                }
                else {
                    inter_on_on = 0;
                }

                if (inter_bin_encoding >= 4) {
                    inter_on_off = 1;
                    inter_bin_encoding -= 4;
                }
                else {
                    inter_on_off = 0;
                }

                if (inter_bin_encoding >= 2) {
                    inter_off_on = 1;
                    inter_bin_encoding -= 2;
                }
                else {
                    inter_off_on = 0;
                }

                if (inter_bin_encoding == 1) inter_off_off = 1;
                else inter_off_off = 0;

                temp_i_value = 0.5 * (sqrt(double(inter_index) * 8.0 + 1.0) + 1.0);
                temp_i_value = floor(temp_i_value);

                lit_i_index = int(temp_i_value);

                lit_j_index = inter_index - ((lit_i_index - 1) * lit_i_index) / 2;

                value_i = optimized_variant[lit_i_index];
                value_j = optimized_variant[lit_j_index];

                if (value_i == 0 && value_j == 0 && inter_off_off == 1) {
                    inter_off_off = 0;
                    atomicSub(&uncovered[lit_i_index - 1], 1);
                }
                else if (value_i == 0 && value_j == 1 && inter_off_on == 1) {
                    inter_off_on = 0;
                    atomicSub(&uncovered[lit_i_index - 1], 1);
                }
                else if (value_i == 1 && value_j == 0 && inter_on_off == 1) {
                    inter_on_off = 0;
                    atomicSub(&uncovered[lit_i_index - 1], 1);
                }
                else if (value_i == 1 && value_j == 1 && inter_on_on == 1) {
                    inter_on_on = 0;
                    atomicSub(&uncovered[lit_i_index - 1], 1);
                }

                if (inter_off_off == 0 && inter_off_on == 0 && inter_on_off == 0 && inter_on_on == 0) {
                    feature_interactions[inter_index] = 0;
                }
                else {
                    inter_bin_encoding = inter_off_off + 2 * inter_off_on + 4 * inter_on_off + 8 * inter_on_on;

                    feature_interactions[inter_index] = inter_bin_encoding;
                }
            }

            loop_index++;
        }
    }
}


__global__
void calculate_feature_probability(int feature_count, int* current_sample_size, char* current_sample, int* one_counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        char value;
        one_counts[index] = 0;

        for (int sample = 0; sample < current_sample_size[0]; sample++) {
            value = current_sample[index + feature_count * sample];

            if (value == 1) one_counts[index]++;
        }
    }
}


__global__
void init_sample(curandState* state, int feature_count, int* current_sample_size, int sampled_variants_size, char* sampled_variants, int* one_counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        int one_count = one_counts[index];
        float prob;
        float prob_threshold = float(one_count) / float(current_sample_size[0]);

        for (int sample = 0; sample < sampled_variants_size; sample++) {

            prob = curand_uniform(&(state[index]));

            if (prob < prob_threshold) sampled_variants[index + feature_count * sample] = 0;
            else sampled_variants[index + feature_count * sample] = 1;
        }
    }
}


__global__
void reset_novel_fi_counts(int sampled_variants_size, unsigned long long* novel_fi_counts) {
    for (int i = 0; i < sampled_variants_size; i++) {
        novel_fi_counts[i] = 0;
    }
}


__global__
void reset_novel_fi_count(unsigned long long* novel_fi_count) {
    novel_fi_count[0] = 0;
}


__global__
void calculate_sample_gain_all(int parallel_limit, int feature_count, unsigned long long feature_interaction_count, int sampled_variants_size, char* sampled_variants, unsigned long long* novel_fi_counts, char* feature_interactions) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;

        unsigned long long inter_index;

        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;
        char value_i, value_j;

        int lit_i_index, lit_j_index;

        double temp_i_value;

        while (loop_index * parallel_limit + index < feature_interaction_count) {

            inter_index = loop_index * parallel_limit + index;

            inter_bin_encoding = feature_interactions[inter_index];

            if (inter_bin_encoding != 0) {
                if (inter_bin_encoding >= 8) {
                    inter_on_on = 1;
                    inter_bin_encoding -= 8;
                }
                else {
                    inter_on_on = 0;
                }

                if (inter_bin_encoding >= 4) {
                    inter_on_off = 1;
                    inter_bin_encoding -= 4;
                }
                else {
                    inter_on_off = 0;
                }

                if (inter_bin_encoding >= 2) {
                    inter_off_on = 1;
                    inter_bin_encoding -= 2;
                }
                else {
                    inter_off_on = 0;
                }

                if (inter_bin_encoding == 1) inter_off_off = 1;
                else inter_off_off = 0;

                temp_i_value = 0.5 * (sqrt(double(inter_index) * 8.0 + 1.0) + 1.0);
                temp_i_value = floor(temp_i_value);

                lit_i_index = int(temp_i_value);

                lit_j_index = inter_index - ((lit_i_index - 1) * lit_i_index) / 2;

                for (int sample = 0; sample < sampled_variants_size; sample++) {
                    value_i = sampled_variants[lit_i_index + feature_count * sample];
                    value_j = sampled_variants[lit_j_index + feature_count * sample];

                    if (value_i == 0 && value_j == 0 && inter_off_off == 1) atomicAdd(&novel_fi_counts[sample], 1);
                    else if (value_i == 0 && value_j == 1 && inter_off_on == 1) atomicAdd(&novel_fi_counts[sample], 1);
                    else if (value_i == 1 && value_j == 0 && inter_on_off == 1) atomicAdd(&novel_fi_counts[sample], 1);
                    else if (value_i == 1 && value_j == 1 && inter_on_on == 1) atomicAdd(&novel_fi_counts[sample], 1);
                }
            }

            loop_index++;
        }
    }
}


__global__
void calculate_sample_gain_optimized(int parallel_limit, int feature_count, unsigned long long feature_interaction_count, char* optimized_variant, unsigned long long* novel_fi_count, char* feature_interactions) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;

        unsigned long long inter_index;

        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;
        char value_i, value_j;

        int lit_i_index, lit_j_index;

        double temp_i_value;

        while (loop_index * parallel_limit + index < feature_interaction_count) {

            inter_index = loop_index * parallel_limit + index;

            inter_bin_encoding = feature_interactions[inter_index];

            if (inter_bin_encoding != 0) {
                if (inter_bin_encoding >= 8) {
                    inter_on_on = 1;
                    inter_bin_encoding -= 8;
                }
                else {
                    inter_on_on = 0;
                }

                if (inter_bin_encoding >= 4) {
                    inter_on_off = 1;
                    inter_bin_encoding -= 4;
                }
                else {
                    inter_on_off = 0;
                }

                if (inter_bin_encoding >= 2) {
                    inter_off_on = 1;
                    inter_bin_encoding -= 2;
                }
                else {
                    inter_off_on = 0;
                }

                if (inter_bin_encoding == 1) inter_off_off = 1;
                else inter_off_off = 0;

                temp_i_value = 0.5 * (sqrt(double(inter_index) * 8.0 + 1.0) + 1.0);
                temp_i_value = floor(temp_i_value);

                lit_i_index = int(temp_i_value);

                lit_j_index = inter_index - ((lit_i_index - 1) * lit_i_index) / 2;

                value_i = optimized_variant[lit_i_index];
                value_j = optimized_variant[lit_j_index];

                if (value_i == 0 && value_j == 0 && inter_off_off == 1) atomicAdd(&novel_fi_count[0], 1);
                else if (value_i == 0 && value_j == 1 && inter_off_on == 1) atomicAdd(&novel_fi_count[0], 1);
                else if (value_i == 1 && value_j == 0 && inter_on_off == 1) atomicAdd(&novel_fi_count[0], 1);
                else if (value_i == 1 && value_j == 1 && inter_on_on == 1) atomicAdd(&novel_fi_count[0], 1);
            }

            loop_index++;
        }
    }
}


__global__
void find_max_sample(int sampled_variants_size, unsigned long long* novel_fi_counts, int* max_index) {
    max_index[0] = -1;
    unsigned long long temp_count, max_count = 0;

    for (int i = 0; i < sampled_variants_size; i++) {
        temp_count = novel_fi_counts[i];
        if (temp_count >= max_count) {
            max_count = temp_count;
            max_index[0] = i;
        }
    }
}


__global__
void write_max_sample_to_optimized(int feature_count, char* sampled_variants, char* optimized_variant, int* max_index) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        optimized_variant[index] = sampled_variants[feature_count * max_index[0] + index];
    }
}


__global__
void write_selected_sample_to_optimized(int feature_count, char* current_sample, char* optimized_variant, int* selected_index) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        optimized_variant[index] = current_sample[feature_count * selected_index[0] + index];
    }
}


__global__
void reset_M_values(int feature_count, int* M_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        M_values[index] = 0;
    }
}


__global__
void calculate_init_M_values(int parallel_limit, int feature_count, char* feature_interactions, char* optimized_variant, int* M_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;

        unsigned long long inter_index;

        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;

        char value;

        int lit_i_index, lit_j_index;

        while (loop_index * parallel_limit + index < feature_count * feature_count) {

            inter_index = loop_index * parallel_limit + index;

            lit_j_index = inter_index % feature_count;
            lit_i_index = (inter_index - lit_j_index) / feature_count;

            if (lit_i_index != lit_j_index) {

                if (lit_j_index < lit_i_index) {
                    inter_index = ((lit_i_index - 1) * lit_i_index) / 2 + lit_j_index;
                }
                else if (lit_j_index > lit_i_index) {
                    inter_index = ((lit_j_index - 1) * lit_j_index) / 2 + lit_i_index;
                }

                inter_bin_encoding = feature_interactions[inter_index];

                if (inter_bin_encoding != 0) {
                    if (inter_bin_encoding >= 8) {
                        inter_on_on = 1;
                        inter_bin_encoding -= 8;
                    }
                    else {
                        inter_on_on = 0;
                    }

                    if (inter_bin_encoding >= 4) {
                        inter_on_off = 1;
                        inter_bin_encoding -= 4;
                    }
                    else {
                        inter_on_off = 0;
                    }

                    if (inter_bin_encoding >= 2) {
                        inter_off_on = 1;
                        inter_bin_encoding -= 2;
                    }
                    else {
                        inter_off_on = 0;
                    }

                    if (inter_bin_encoding == 1) inter_off_off = 1;
                    else inter_off_off = 0;

                    value = optimized_variant[lit_j_index];

                    if (lit_j_index < lit_i_index) {
                        if (inter_on_on == 1) atomicAdd(&M_values[lit_i_index], value);
                        if (inter_on_off == 1) atomicAdd(&M_values[lit_i_index], -value + 1);
                        if (inter_off_on == 1) atomicAdd(&M_values[lit_i_index], -value);
                        if (inter_off_off == 1) atomicAdd(&M_values[lit_i_index], value - 1);
                    }
                    else if (lit_j_index > lit_i_index) {
                        if (inter_on_on == 1) atomicAdd(&M_values[lit_i_index], value);
                        if (inter_on_off == 1) atomicAdd(&M_values[lit_i_index], -value);
                        if (inter_off_on == 1) atomicAdd(&M_values[lit_i_index], -value + 1);
                        if (inter_off_off == 1) atomicAdd(&M_values[lit_i_index], value - 1);
                    }
                }
            }

            loop_index++;
        }
    }
}


__global__
void reset_flip_counts(int feature_count, int* is_flip_valid) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        is_flip_valid[index] = 0;
    }
}


__global__
void reset_dual_flip_counts(int parallel_limit, int feature_count, unsigned long long feature_interaction_count, int* is_flip_valid, int* dual_invalids, int* dual_valids, int* valid_thresholds) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int loop_index = 0;

    if (index < feature_count) {
        is_flip_valid[index] = 0;
        // valid_thresholds[index] = 0;
    }

    /*
    if (index < parallel_limit) {
        while (loop_index * parallel_limit + index < feature_interaction_count) {
            dual_invalids[loop_index * parallel_limit + index] = 0;
            dual_valids[loop_index * parallel_limit + index] = 0;

            loop_index++;
        }
    }
    */
}


__global__
void calculate_flip_validity_optimized(int parallel_limit, int feature_count, unsigned long long total_lits, short* current_clause_values, short* min_clause_values, int* feat_to_clause_reference, int* clause_to_feat_reference, char* optimized_variant, int* is_flip_valid) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;
        int clause_index, lit_index;
        bool is_negative;
        char value_i;

        while (loop_index * parallel_limit + index < total_lits) {

            clause_index = feat_to_clause_reference[loop_index * parallel_limit + index];
            lit_index = clause_to_feat_reference[loop_index * parallel_limit + index];

            value_i = optimized_variant[lit_index];

            if (clause_index < 0) {
                clause_index = -clause_index - 1;
                is_negative = true;
            }
            else {
                clause_index = clause_index - 1;
                is_negative = false;
            }

            if (current_clause_values[clause_index] == min_clause_values[clause_index]) {
                if (is_negative) {
                    if (value_i == 0) atomicAdd(&is_flip_valid[lit_index], 1);
                }
                else {
                    if (value_i == 1) atomicAdd(&is_flip_valid[lit_index], 1);
                }
            }

            loop_index++;
        }
    }
}


__global__
void calculate_dual_flip_validity_optimized(int parallel_limit, int feature_count, unsigned long long total_lits, short* current_clause_values, short* min_clause_values, int* feat_to_clause_reference, int* clause_to_feat_reference, int* clause_list, short* clause_list_sizes, int lits_in_clause_limit, char* optimized_variant, int* is_flip_valid, int* dual_invalids, int* dual_valids, int* valid_thresholds) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {

        int loop_index = 0;
        int clause_index, lit_i_index, lit_j_index, delta_value, delta_value_temp;
        bool is_i_negative, is_j_negative, is_invalid;
        char value_i, value_j;

        unsigned long long inter_index;

        while (loop_index * parallel_limit + index < total_lits) {

            clause_index = feat_to_clause_reference[loop_index * parallel_limit + index];
            lit_i_index = clause_to_feat_reference[loop_index * parallel_limit + index];

            value_i = optimized_variant[lit_i_index];

            if (clause_index < 0) {
                clause_index = -clause_index - 1;
                is_i_negative = true;
            }
            else {
                clause_index = clause_index - 1;
                is_i_negative = false;
            }

            if (is_i_negative) {
                if (value_i == 0) delta_value = -1;
                else delta_value = 1;
            }
            else {
                if (value_i == 1) delta_value = -1;
                else delta_value = 1;
            }

            if (current_clause_values[clause_index] + delta_value < min_clause_values[clause_index]) {
                // atomicAdd(&valid_thresholds[lit_i_index], 1);
                atomicAdd(&is_flip_valid[lit_i_index], 1);
                // is_invalid = true;
            }
            /*
            else {
                is_invalid = false;
            }
            */

            /*
            if (lit_i_index > 0) {

                for (int i = 0; i < clause_list_sizes[clause_index]; i++) {
                    lit_j_index = clause_list[clause_index * lits_in_clause_limit + i];

                    if (lit_j_index < 0) {
                        lit_j_index = -lit_j_index - 1;
                        is_j_negative = true;
                    }
                    else {
                        lit_j_index = lit_j_index - 1;
                        is_j_negative = false;
                    }

                    if (lit_j_index < lit_i_index) {
                        value_j = optimized_variant[lit_j_index];

                        if (is_j_negative) {
                            if (value_j == 0) delta_value_temp = delta_value - 1;
                            else delta_value_temp = delta_value + 1;
                        }
                        else {
                            if (value_j == 1) delta_value_temp = delta_value - 1;
                            else delta_value_temp = delta_value + 1;
                        }

                        if (!is_invalid) {
                            if (current_clause_values[clause_index] + delta_value_temp < min_clause_values[clause_index]) {
                                inter_index = ((lit_i_index - 1) * lit_i_index) / 2 + lit_j_index;
                                atomicAdd(&dual_invalids[inter_index], 1);
                            }
                        }
                        else {
                            if (current_clause_values[clause_index] + delta_value_temp >= min_clause_values[clause_index]) {
                                inter_index = ((lit_i_index - 1) * lit_i_index) / 2 + lit_j_index;
                                atomicAdd(&dual_valids[inter_index], 1);
                            }
                        }
                    }
                }
            }
            */

            loop_index++;

        }
    }
}


__global__
void calculate_max_gain_dual_with_validity_check(int parallel_limit, int feature_count, unsigned long long feature_interaction_count, char* feature_interactions, char* optimized_variant, int* M_values, int* max_indices_i_m, int* max_indices_j_m, int* max_gains_m, int step_size, int* current_step, int* is_flip_valid, int* dual_invalids, int* dual_valids, int* valid_thresholds, char* are_feature_pairs_dependent) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int loop_index = 0;

    if (index < parallel_limit) {
        unsigned long long inter_index;

        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;

        int M_value_i, M_value_j, M_value_i_reduced, M_value_j_reduced, gain_value, q_value;

        char value_i, value_j;

        int lit_i_index, lit_j_index;

        double temp_i_value;

        bool valid;

        max_indices_i_m[index] = -1;
        max_indices_j_m[index] = -1;
        max_gains_m[index] = 0;

        while (loop_index * parallel_limit + index < feature_interaction_count) {

            inter_index = loop_index * parallel_limit + index;

            temp_i_value = 0.5 * (sqrt(double(inter_index) * 8.0 + 1.0) + 1.0);
            temp_i_value = floor(temp_i_value);

            lit_i_index = int(temp_i_value);

            lit_j_index = inter_index - ((lit_i_index - 1) * lit_i_index) / 2;

            if (lit_j_index % step_size == current_step[0]) {

                value_i = optimized_variant[lit_i_index];
                M_value_i = M_values[lit_i_index];

                value_j = optimized_variant[lit_j_index];
                M_value_j = M_values[lit_j_index];

                if (value_j == 0) {
                    if (is_flip_valid[lit_i_index] == 0) {
                        gain_value = M_value_i * (1 - 2 * value_i);

                        if (gain_value > max_gains_m[index]) {
                            max_gains_m[index] = gain_value;
                            max_indices_i_m[index] = lit_i_index;
                            max_indices_j_m[index] = -1;
                        }
                    }
                }

                valid = false;

                if (are_feature_pairs_dependent[inter_index] == 0) {
                    if (is_flip_valid[lit_i_index] == 0 && is_flip_valid[lit_j_index] == 0) valid = true;
                }
                /*
                else {
                    if (dual_invalids[inter_index] == 0) valid = true;

                    if (valid_thresholds[index] > 0) {
                        if (dual_valids[inter_index] < valid_thresholds[index]) valid = false;
                    }
                }
                */

                if (valid) {

                    inter_bin_encoding = feature_interactions[inter_index];

                    if (inter_bin_encoding != 0) {

                        if (inter_bin_encoding >= 8) {
                            inter_on_on = 1;
                            inter_bin_encoding -= 8;
                        }
                        else {
                            inter_on_on = 0;
                        }

                        if (inter_bin_encoding >= 4) {
                            inter_on_off = 1;
                            inter_bin_encoding -= 4;
                        }
                        else {
                            inter_on_off = 0;
                        }

                        if (inter_bin_encoding >= 2) {
                            inter_off_on = 1;
                            inter_bin_encoding -= 2;
                        }
                        else {
                            inter_off_on = 0;
                        }

                        if (inter_bin_encoding == 1) inter_off_off = 1;
                        else inter_off_off = 0;

                        q_value = 0;

                        if (inter_on_on == 1) q_value += 1;
                        if (inter_on_off == 1) q_value += -1;
                        if (inter_off_on == 1) q_value += -1;
                        if (inter_off_off == 1) q_value += 1;

                        M_value_i_reduced = M_value_i - q_value * value_j;
                        M_value_j_reduced = M_value_j - q_value * value_i;

                        gain_value = M_value_i_reduced * (1 - 2 * value_i) + M_value_j_reduced * (1 - 2 * value_j);
                        if (q_value != 0) gain_value += q_value * (-value_i * value_j + (1 - value_i) * (1 - value_j));

                    }
                    else {
                        gain_value = M_value_i * (1 - 2 * value_i) + M_value_j * (1 - 2 * value_j);
                    }

                    if (gain_value > max_gains_m[index]) {
                        max_gains_m[index] = gain_value;
                        max_indices_i_m[index] = lit_i_index;
                        max_indices_j_m[index] = lit_j_index;
                    }
                }

            }

            loop_index++;
        }
    }
}


__global__
void calculate_max_gain_dual_with_validity_check_2(int feature_count, char* feature_interactions, char* optimized_variant, int* M_values, int* max_indices_j_m, int* max_gains_m, int step_size, int* current_step, int* is_flip_valid, int* dual_invalids, int* dual_valids, int* valid_thresholds, char* are_feature_pairs_dependent) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {

        unsigned long long inter_index;
        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;

        int M_value_i, M_value_j, M_value_i_reduced, M_value_j_reduced, gain_value, q_value;
        char value_i = optimized_variant[index], value_j;
        int lit_value;

        max_indices_j_m[index] = -1;
        max_gains_m[index] = 0;

        M_value_i = M_values[index];

        if (is_flip_valid[index] == 0) {
            gain_value = M_value_i * (1 - 2 * value_i);

            if (gain_value > max_gains_m[index]) {
                max_gains_m[index] = gain_value;
                max_indices_j_m[index] = -1;
            }

            for (int j = current_step[0]; j < index; j += step_size) {

                if (is_flip_valid[j] == 0) {
                    value_j = optimized_variant[j];

                    M_value_j = M_values[j];

                    inter_index = ((index - 1) * index) / 2 + j;

                    if (are_feature_pairs_dependent[inter_index] == 0) {

                        inter_bin_encoding = feature_interactions[inter_index];

                        if (inter_bin_encoding != 0) {

                            if (inter_bin_encoding >= 8) {
                                inter_on_on = 1;
                                inter_bin_encoding -= 8;
                            }
                            else {
                                inter_on_on = 0;
                            }

                            if (inter_bin_encoding >= 4) {
                                inter_on_off = 1;
                                inter_bin_encoding -= 4;
                            }
                            else {
                                inter_on_off = 0;
                            }

                            if (inter_bin_encoding >= 2) {
                                inter_off_on = 1;
                                inter_bin_encoding -= 2;
                            }
                            else {
                                inter_off_on = 0;
                            }

                            if (inter_bin_encoding == 1) inter_off_off = 1;
                            else inter_off_off = 0;

                            q_value = 0;

                            if (inter_on_on == 1) q_value += 1;
                            if (inter_on_off == 1) q_value += -1;
                            if (inter_off_on == 1) q_value += -1;
                            if (inter_off_off == 1) q_value += 1;

                            M_value_i_reduced = M_value_i - q_value * value_j;
                            M_value_j_reduced = M_value_j - q_value * value_i;

                            gain_value = M_value_i_reduced * (1 - 2 * value_i) + M_value_j_reduced * (1 - 2 * value_j);
                            if (q_value != 0) gain_value += q_value * (-value_i * value_j + (1 - value_i) * (1 - value_j));

                        }
                        else {
                            gain_value = M_value_i * (1 - 2 * value_i) + M_value_j * (1 - 2 * value_j);

                        }

                        if (gain_value > max_gains_m[index]) {
                            max_gains_m[index] = gain_value;
                            max_indices_j_m[index] = j;
                        }
                    }
                }
            }
        }
    }
}


__global__
void preprocess_maximum(int parallel_limit, int batch_size, int* max_indices_i_m, int* max_indices_j_m, int* max_gains_m, int* max_indices_i_reduced, int* max_indices_j_reduced, int* max_gains_reduced) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < parallel_limit) {
        max_gains_reduced[index] = 0;
        max_indices_i_reduced[index] = -1;
        max_indices_j_reduced[index] = -1;

        int iterator;

        for (int i = 0; i < batch_size; i++) {
            iterator = index * batch_size + i;

            if (max_gains_m[iterator] > max_gains_reduced[index]) {
                max_gains_reduced[index] = max_gains_m[iterator];
                max_indices_i_reduced[index] = max_indices_i_m[iterator];
                max_indices_j_reduced[index] = max_indices_j_m[iterator];
            }
        }
    }
}


__global__
void adapt_M_values(int feature_count, char* feature_interactions, char* optimized_variant, int* M_values, int* max_index_m) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        unsigned long long inter_index;
        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;

        char value;

        int q_value = 0;

        if (max_index_m[0] != index) {

            if (max_index_m[0] < index) {
                inter_index = ((index - 1) * index) / 2 + max_index_m[0];
            }
            else if (max_index_m[0] > index) {
                inter_index = ((max_index_m[0] - 1) * max_index_m[0]) / 2 + index;
            }

            inter_bin_encoding = feature_interactions[inter_index];

            if (inter_bin_encoding != 0) {
                if (inter_bin_encoding >= 8) {
                    inter_on_on = 1;
                    inter_bin_encoding -= 8;
                }
                else {
                    inter_on_on = 0;
                }

                if (inter_bin_encoding >= 4) {
                    inter_on_off = 1;
                    inter_bin_encoding -= 4;
                }
                else {
                    inter_on_off = 0;
                }

                if (inter_bin_encoding >= 2) {
                    inter_off_on = 1;
                    inter_bin_encoding -= 2;
                }
                else {
                    inter_off_on = 0;
                }

                if (inter_bin_encoding == 1) inter_off_off = 1;
                else inter_off_off = 0;

                if (inter_on_on == 1) q_value += 1;
                if (inter_on_off == 1) q_value += -1;
                if (inter_off_on == 1) q_value += -1;
                if (inter_off_off == 1) q_value += 1;

                if (q_value != 0) {
                    value = optimized_variant[max_index_m[0]];
                    M_values[index] += q_value * (1 - 2 * value);
                }
            }
        }
    }
}


__global__
void adapt_M_values_dual(int feature_count, char* feature_interactions, char* optimized_variant, int* M_values, int* max_index_i_m, int* max_index_j_m) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        unsigned long long inter_index;
        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char inter_bin_encoding;

        char value_i = optimized_variant[max_index_i_m[0]];
        char value_j = optimized_variant[max_index_j_m[0]];

        int q_value = 0;

        if (max_index_i_m[0] != index) {

            if (max_index_i_m[0] < index) {
                inter_index = ((index - 1) * index) / 2 + max_index_i_m[0];
            }
            else if (max_index_i_m[0] > index) {
                inter_index = ((max_index_i_m[0] - 1) * max_index_i_m[0]) / 2 + index;
            }

            inter_bin_encoding = feature_interactions[inter_index];

            if (inter_bin_encoding != 0) {
                if (inter_bin_encoding >= 8) {
                    inter_on_on = 1;
                    inter_bin_encoding -= 8;
                }
                else {
                    inter_on_on = 0;
                }

                if (inter_bin_encoding >= 4) {
                    inter_on_off = 1;
                    inter_bin_encoding -= 4;
                }
                else {
                    inter_on_off = 0;
                }

                if (inter_bin_encoding >= 2) {
                    inter_off_on = 1;
                    inter_bin_encoding -= 2;
                }
                else {
                    inter_off_on = 0;
                }

                if (inter_bin_encoding == 1) inter_off_off = 1;
                else inter_off_off = 0;

                if (inter_on_on == 1) q_value += 1;
                if (inter_on_off == 1) q_value += -1;
                if (inter_off_on == 1) q_value += -1;
                if (inter_off_off == 1) q_value += 1;

                if (q_value != 0) {
                    M_values[index] += q_value * (1 - 2 * value_i);
                }
            }
        }

        if (max_index_j_m[0] != -1) {
            q_value = 0;

            if (max_index_j_m[0] != index) {

                if (max_index_j_m[0] < index) {
                    inter_index = ((index - 1) * index) / 2 + max_index_j_m[0];
                }
                else if (max_index_j_m[0] > index) {
                    inter_index = ((max_index_j_m[0] - 1) * max_index_j_m[0]) / 2 + index;
                }

                inter_bin_encoding = feature_interactions[inter_index];

                if (inter_bin_encoding != 0) {
                    if (inter_bin_encoding >= 8) {
                        inter_on_on = 1;
                        inter_bin_encoding -= 8;
                    }
                    else {
                        inter_on_on = 0;
                    }

                    if (inter_bin_encoding >= 4) {
                        inter_on_off = 1;
                        inter_bin_encoding -= 4;
                    }
                    else {
                        inter_on_off = 0;
                    }

                    if (inter_bin_encoding >= 2) {
                        inter_off_on = 1;
                        inter_bin_encoding -= 2;
                    }
                    else {
                        inter_off_on = 0;
                    }

                    if (inter_bin_encoding == 1) inter_off_off = 1;
                    else inter_off_off = 0;

                    if (inter_on_on == 1) q_value += 1;
                    if (inter_on_off == 1) q_value += -1;
                    if (inter_off_on == 1) q_value += -1;
                    if (inter_off_off == 1) q_value += 1;

                    if (q_value != 0) {
                        M_values[index] += q_value * (1 - 2 * value_j);
                    }
                }
            }
        }
    }
}


__global__
void adapt_clause_values(short* current_clause_values, int* clauses_with_lit, int* feat_to_clause_reference, unsigned long long* reference_start_points, char* optimized_variant, int* max_index_m) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < clauses_with_lit[max_index_m[0]]) {

        int clause_index = feat_to_clause_reference[reference_start_points[max_index_m[0]] + index];
        if (clause_index < 0) {
            clause_index = -clause_index - 1;

            if (optimized_variant[max_index_m[0]] == 0) {
                current_clause_values[clause_index] -= 1;
            }
            else {
                current_clause_values[clause_index] += 1;
            }
        }
        else {
            clause_index = clause_index - 1;

            if (optimized_variant[max_index_m[0]] == 0) {
                current_clause_values[clause_index] += 1;
            }
            else {
                current_clause_values[clause_index] -= 1;
            }
        }
    }
}


__global__
void optimize_variant(char* optimized_variant, int* max_index_m) {
    optimized_variant[max_index_m[0]] = 1 - optimized_variant[max_index_m[0]];
}


__global__
void optimize_variant_dual(char* optimized_variant, int* max_index_i_m, int* max_index_j_m) {
    if (max_index_j_m[0] == -1) {
        optimized_variant[max_index_i_m[0]] = 1 - optimized_variant[max_index_i_m[0]];
    }
    else {
        optimized_variant[max_index_i_m[0]] = 1 - optimized_variant[max_index_i_m[0]];
        optimized_variant[max_index_j_m[0]] = 1 - optimized_variant[max_index_j_m[0]];
    }
}


__global__
void write_optimized_to_current_sample(int feature_count, char* optimized_variant, int* insertion_point, char* current_sample) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < feature_count) {
        current_sample[feature_count * insertion_point[0] + index] = optimized_variant[index];
    }
}


__global__
void increase_current_sample_size(int* current_sample_size) {
    current_sample_size[0] += 1;
}


int main(void)
{
    setUsageHelp("USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");

    IntOption    verb("MAIN", "verb", "Verbosity level (0=silent, 1=some, 2=more).", 0, IntRange(0, 2));
    BoolOption   pre("MAIN", "pre", "Completely turn on/off any preprocessing.", false);
    StringOption dimacs("MAIN", "dimacs", "If given, stop after preprocessing and write the result to this file.");
    IntOption    cpu_lim("MAIN", "cpu-lim", "Limit on CPU time allowed in seconds.\n", INT32_MAX, IntRange(0, INT32_MAX));
    IntOption    mem_lim("MAIN", "mem-lim", "Limit on memory usage in megabytes.\n", INT32_MAX, IntRange(0, INT32_MAX));

    std::mt19937_64 rng;
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
    rng.seed(ss);
    std::uniform_real_distribution<double> unif(0, 1);
    std::uniform_real_distribution<double> unif2(0, 10000);
    std::uniform_real_distribution<double> unif3(0, 10000);

    vector<int> cnf_command_params;
    vec<Lit> lits;

    char* active_literals = nullptr, * inactive_literals = nullptr;
    int* clause_list = nullptr, * clause_list_c;
    short* clause_list_sizes = nullptr, * clause_list_sizes_c;
    short* current_clause_values_c, * min_clause_values_c;

    int* feat_to_clause_reference = nullptr, * feat_to_clause_reference_c;
    int* clause_to_feat_reference = nullptr, * clause_to_feat_reference_c;
    int* clauses_with_lit = nullptr, * clauses_with_lit_c, * current_reference_iterators = nullptr;
    unsigned long long* reference_start_points = nullptr, * reference_start_points_c;

    int* dual_invalids_c, int* dual_valids_c;
    int* valid_thresholds_c;
    char* are_feature_pairs_dependent, * are_feature_pairs_dependent_c;

    char* invalid_feature_interactions;
    int* invalids_uncovered;

    // ---------------------------
    const string cnf_file_name = "C:\\Users\\lenna\\Downloads\\bmc\\automotive.cnf";
    const bool validity_checking = true;
    const int lits_in_clause_limit = 25;
    const int sampled_variants_size = 100;
    // ---------------------------

    ifstream cnf_file(cnf_file_name);

    int feature_count = -1, init_clause_count, reduced_clause_count = 0;
    int feature_interaction_count; 

    string file_line;
    int ascii_value, lit_value, char_counter;
    bool is_lit_negative, is_comment, is_command;

    int max_lits_per_clause = 0;
    unsigned long long total_lits_in_file = 0;

    SimpSolver* S = new SimpSolver[sampled_variants_size];

    for (int i = 0; i < sampled_variants_size; i++) {
        S[i].eliminate(true);
        S[i].verbosity = verb;
        S[i].setRandomSeed(unif3(rng));
    }

    auto start0 = std::chrono::high_resolution_clock::now();

    if (cnf_file.is_open()) {
        while (getline(cnf_file, file_line)) {

            if (validity_checking) {
                if (feature_count != -1) {
                    for (int i = 0; i < feature_count; i++) {
                        inactive_literals[i] = 0;
                        active_literals[i] = 0;
                    }
                }
            }

            lit_value = 0;

            lits.clear();

            is_lit_negative = false;
            is_comment = false;
            is_command = false;

            char_counter = 0;

            for (char& c : file_line) {
                ascii_value = int(c);
                if (char_counter == 0 && ascii_value == 99) {
                    is_comment = true;
                    break;
                }

                if (char_counter == 0 && ascii_value == 112) is_command = true;

                if (ascii_value >= 48 && ascii_value <= 57) {
                    if (lit_value == 0) {
                        lit_value = ascii_value - 48;
                    }
                    else {
                        lit_value *= 10;
                        lit_value += ascii_value - 48;
                    }
                }
                else if (ascii_value == 45) {
                    is_lit_negative = true;
                }
                else {
                    if (!is_command) {
                        if (lit_value > 0) {
                            if (validity_checking) {
                                if (is_lit_negative) inactive_literals[lit_value - 1] = 1;
                                else active_literals[lit_value - 1] = 1;
                            }
                            else {
                                if (is_lit_negative) {
                                    lits.push(~mkLit(lit_value - 1));
                                    clause_list[reduced_clause_count * lits_in_clause_limit + clause_list_sizes[reduced_clause_count]] = -lit_value;
                                    clause_list_sizes[reduced_clause_count] += 1;
                                }
                                else {
                                    lits.push(mkLit(lit_value - 1));
                                    clause_list[reduced_clause_count * lits_in_clause_limit + clause_list_sizes[reduced_clause_count]] = lit_value;
                                    clause_list_sizes[reduced_clause_count] += 1;
                                }
                            }
                        }
                    }
                    else {
                        if (lit_value > 0) {
                            cnf_command_params.push_back(lit_value);
                        }
                    }

                    lit_value = 0;
                    is_lit_negative = false;
                }

                char_counter++;
            }

            if (is_comment) continue;

            if (!is_command) {
                if (lit_value > 0) {
                    if (validity_checking) {
                        if (is_lit_negative) inactive_literals[lit_value - 1] = 1;
                        else active_literals[lit_value - 1] = 1;
                    }
                    else {
                        if (is_lit_negative) {
                            lits.push(~mkLit(lit_value - 1));
                            clause_list[reduced_clause_count * lits_in_clause_limit + clause_list_sizes[reduced_clause_count]] = -lit_value;
                            clause_list_sizes[reduced_clause_count] += 1;
                        }
                        else {
                            lits.push(mkLit(lit_value - 1));
                            clause_list[reduced_clause_count * lits_in_clause_limit + clause_list_sizes[reduced_clause_count]] = lit_value;
                            clause_list_sizes[reduced_clause_count] += 1;
                        }
                    }
                }

                bool irrelevant_clause = false;

                if (validity_checking) {
                    for (int n = 0; n < feature_count; n++) {
                        if (active_literals[n] == 1 && inactive_literals[n] == 1) {
                            irrelevant_clause = true;
                            break;
                        }
                    }
                }

                if (!irrelevant_clause) {
                    if (validity_checking) {
                        for (int n = 0; n < feature_count; n++) {
                            if (inactive_literals[n] == 1) {
                                lits.push(~mkLit(n));
                                clause_list[reduced_clause_count * lits_in_clause_limit + clause_list_sizes[reduced_clause_count]] = -(n + 1);
                                clause_list_sizes[reduced_clause_count] += 1;
                            }
                            else if (active_literals[n] == 1) {
                                lits.push(mkLit(n));
                                clause_list[reduced_clause_count * lits_in_clause_limit + clause_list_sizes[reduced_clause_count]] = (n + 1);
                                clause_list_sizes[reduced_clause_count] += 1;
                            }
                        }
                    }

                    if (clause_list_sizes[reduced_clause_count] == 1) {
                        int lit1;
                        bool is_1_negative;
                        unsigned long long inter_index;
                        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
                        char inter_bin_encoding;
                        int old_invalids_count, new_invalids_count;

                        lit1 = clause_list[reduced_clause_count * lits_in_clause_limit];

                        if (lit1 < 0) {
                            lit1 = -lit1 - 1;
                            is_1_negative = true;
                        }
                        else {
                            lit1 = lit1 - 1;
                            is_1_negative = false;
                        }

                        for (int lit2 = 0; lit2 < feature_count; lit2++) {
                            if (lit1 != lit2) {

                                if (lit1 < lit2) inter_index = ((lit2 - 1) * lit2) / 2 + lit1;
                                else inter_index = ((lit1 - 1) * lit1) / 2 + lit2;

                                inter_bin_encoding = invalid_feature_interactions[inter_index];

                                if (inter_bin_encoding != 0) {
                                    if (inter_bin_encoding >= 8) {
                                        inter_on_on = 1;
                                        inter_bin_encoding -= 8;
                                    }
                                    else {
                                        inter_on_on = 0;
                                    }

                                    if (inter_bin_encoding >= 4) {
                                        inter_on_off = 1;
                                        inter_bin_encoding -= 4;
                                    }
                                    else {
                                        inter_on_off = 0;
                                    }

                                    if (inter_bin_encoding >= 2) {
                                        inter_off_on = 1;
                                        inter_bin_encoding -= 2;
                                    }
                                    else {
                                        inter_off_on = 0;
                                    }

                                    if (inter_bin_encoding == 1) inter_off_off = 1;
                                    else inter_off_off = 0;

                                    old_invalids_count = inter_on_on + inter_on_off + inter_off_on + inter_off_off;

                                    if (lit1 < lit2) {
                                        if (is_1_negative) {
                                            inter_on_on = 1;
                                            inter_off_on = 1;
                                        }
                                        else {
                                            inter_on_off = 1;
                                            inter_off_off = 1;
                                        }
                                    }
                                    else {
                                        if (is_1_negative) {
                                            inter_on_on = 1;
                                            inter_on_off = 1;
                                        }
                                        else {
                                            inter_off_on = 1;
                                            inter_off_off = 1;
                                        }
                                    }

                                    new_invalids_count = inter_on_on + inter_on_off + inter_off_on + inter_off_off;

                                    if (lit1 < lit2) invalids_uncovered[lit2 - 1] += new_invalids_count - old_invalids_count;
                                    else invalids_uncovered[lit1 - 1] += new_invalids_count - old_invalids_count;

                                    inter_bin_encoding = inter_off_off + 2 * inter_off_on + 4 * inter_on_off + 8 * inter_on_on;

                                    invalid_feature_interactions[inter_index] = inter_bin_encoding;
                                }
                                else {
                                    inter_on_on = 0;
                                    inter_on_off = 0;
                                    inter_off_on = 0;
                                    inter_off_off = 0;

                                    if (lit1 < lit2) {
                                        if (is_1_negative) {
                                            inter_on_on = 1;
                                            inter_off_on = 1;
                                        }
                                        else {
                                            inter_on_off = 1;
                                            inter_off_off = 1;
                                        }
                                    }
                                    else {
                                        if (is_1_negative) {
                                            inter_on_on = 1;
                                            inter_on_off = 1;
                                        }
                                        else {
                                            inter_off_on = 1;
                                            inter_off_off = 1;
                                        }
                                    }

                                    if (lit1 < lit2) invalids_uncovered[lit2 - 1] += 2;
                                    else invalids_uncovered[lit1 - 1] += 2;

                                    inter_bin_encoding = inter_off_off + 2 * inter_off_on + 4 * inter_on_off + 8 * inter_on_on;

                                    invalid_feature_interactions[inter_index] = inter_bin_encoding;
                                }
                            }
                        }
                    }
                    else if (clause_list_sizes[reduced_clause_count] == 2) {
                        int lit1, lit2;
                        bool is_1_negative, is_2_negative;
                        unsigned long long inter_index;
                        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
                        char inter_bin_encoding;
                        int old_invalids_count, new_invalids_count;

                        lit1 = clause_list[reduced_clause_count * lits_in_clause_limit];
                        lit2 = clause_list[reduced_clause_count * lits_in_clause_limit + 1];

                        if (lit1 < 0) {
                            lit1 = -lit1 - 1;
                            is_1_negative = true;
                        }
                        else {
                            lit1 = lit1 - 1;
                            is_1_negative = false;
                        }

                        if (lit2 < 0) {
                            lit2 = -lit2 - 1;
                            is_2_negative = true;
                        }
                        else {
                            lit2 = lit2 - 1;
                            is_2_negative = false;
                        }

                        if (lit1 < lit2) inter_index = ((lit2 - 1) * lit2) / 2 + lit1;
                        else inter_index = ((lit1 - 1) * lit1) / 2 + lit2;

                        inter_bin_encoding = invalid_feature_interactions[inter_index];

                        if (inter_bin_encoding != 0) {
                            if (inter_bin_encoding >= 8) {
                                inter_on_on = 1;
                                inter_bin_encoding -= 8;
                            }
                            else {
                                inter_on_on = 0;
                            }

                            if (inter_bin_encoding >= 4) {
                                inter_on_off = 1;
                                inter_bin_encoding -= 4;
                            }
                            else {
                                inter_on_off = 0;
                            }

                            if (inter_bin_encoding >= 2) {
                                inter_off_on = 1;
                                inter_bin_encoding -= 2;
                            }
                            else {
                                inter_off_on = 0;
                            }

                            if (inter_bin_encoding == 1) inter_off_off = 1;
                            else inter_off_off = 0;

                            old_invalids_count = inter_on_on + inter_on_off + inter_off_on + inter_off_off;

                            if (lit1 < lit2) {
                                if (is_1_negative && is_2_negative) inter_on_on = 1;
                                else if (!is_1_negative && is_2_negative) inter_on_off = 1;
                                else if (is_1_negative && !is_2_negative) inter_off_on = 1;
                                else if (!is_1_negative && !is_2_negative) inter_off_off = 1;
                            }
                            else {
                                if (is_1_negative && is_2_negative) inter_on_on = 1;
                                else if (!is_1_negative && is_2_negative) inter_off_on = 1;
                                else if (is_1_negative && !is_2_negative) inter_on_off = 1;
                                else if (!is_1_negative && !is_2_negative) inter_off_off = 1;
                            }

                            new_invalids_count = inter_on_on + inter_on_off + inter_off_on + inter_off_off;

                            if (lit1 < lit2) invalids_uncovered[lit2 - 1] += new_invalids_count - old_invalids_count;
                            else invalids_uncovered[lit1 - 1] += new_invalids_count - old_invalids_count;

                            inter_bin_encoding = inter_off_off + 2 * inter_off_on + 4 * inter_on_off + 8 * inter_on_on;

                            invalid_feature_interactions[inter_index] = inter_bin_encoding;
                        }
                        else {
                            inter_on_on = 0;
                            inter_on_off = 0;
                            inter_off_on = 0;
                            inter_off_off = 0;

                            if (lit1 < lit2) {
                                if (is_1_negative && is_2_negative) inter_on_on = 1;
                                else if (!is_1_negative && is_2_negative) inter_on_off = 1;
                                else if (is_1_negative && !is_2_negative) inter_off_on = 1;
                                else if (!is_1_negative && !is_2_negative) inter_off_off = 1;
                            }
                            else {
                                if (is_1_negative && is_2_negative) inter_on_on = 1;
                                else if (!is_1_negative && is_2_negative) inter_off_on = 1;
                                else if (is_1_negative && !is_2_negative) inter_on_off = 1;
                                else if (!is_1_negative && !is_2_negative) inter_off_off = 1;
                            }

                            if (lit1 < lit2) invalids_uncovered[lit2 - 1]++;
                            else invalids_uncovered[lit1 - 1]++;

                            inter_bin_encoding = inter_off_off + 2 * inter_off_on + 4 * inter_on_off + 8 * inter_on_on;

                            invalid_feature_interactions[inter_index] = inter_bin_encoding;
                        }
                    }

                    if (clause_list_sizes[reduced_clause_count] > max_lits_per_clause) max_lits_per_clause = clause_list_sizes[reduced_clause_count];

                    if (clause_list_sizes[reduced_clause_count] > lits_in_clause_limit) {
                        cout << "Error: clause has more than " << lits_in_clause_limit << " (max.) literals! Exiting..." << endl;
                        exit(0);
                    }

                    total_lits_in_file += clause_list_sizes[reduced_clause_count];

                    reduced_clause_count++;

                    for (int i = 0; i < sampled_variants_size; i++) S[i].addClause_(lits);

                    if (reduced_clause_count % 5000 == 0) cout << "Loaded clause nr. " << reduced_clause_count << endl;
                }
            }
            else {
                if (lit_value > 0) cnf_command_params.push_back(lit_value);

                feature_count = cnf_command_params[0];
                init_clause_count = cnf_command_params[1];

                cout << "Feature / Init. clause count: " << feature_count << " / " << init_clause_count << endl;

                if (validity_checking) {
                    active_literals = (char*)malloc(feature_count * sizeof(char));
                    inactive_literals = (char*)malloc(feature_count * sizeof(char));

                    for (int i = 0; i < feature_count; i++) {
                        inactive_literals[i] = 0;
                        active_literals[i] = 0;

                        for (int j = 0; j < sampled_variants_size; j++) S[j].newVar();   
                    }
                }
                else {
                    for (int i = 0; i < feature_count; i++) {
                        for (int j = 0; j < sampled_variants_size; j++) S[j].newVar();
                    }
                }

                clause_list = (int*)malloc(init_clause_count * lits_in_clause_limit * sizeof(int));
                clause_list_sizes = (short*)malloc(init_clause_count * sizeof(short));

                feature_interaction_count = (feature_count * (feature_count - 1)) / 2;
                invalid_feature_interactions = (char*)malloc(feature_interaction_count * sizeof(char));
                invalids_uncovered = (int*)malloc((feature_count - 1) * sizeof(int));


                for (int i = 0; i < init_clause_count * lits_in_clause_limit; i++) clause_list[i] = 0;
                for (int i = 0; i < init_clause_count; i++) clause_list_sizes[i] = 0;
                for (int i = 0; i < feature_interaction_count; i++) invalid_feature_interactions[i] = 0;
                for (int i = 0; i < feature_count - 1; i++) invalids_uncovered[i] = 0;
            }
        }

        cnf_file.close();
    }
    else {
        cout << "ERROR! Cannot open file! Exiting..." << endl;
        exit(0);
    }

    cout << "\nReduced clause count: " << reduced_clause_count << endl;
    cout << "Max lits per clause: " << max_lits_per_clause << endl;
    cout << "Mean lits per clause: " << float(total_lits_in_file) / float(reduced_clause_count) << endl;
    
    unsigned long long invalids_total = 0;

    for (int i = 0; i < feature_count - 1; i++) {
        invalids_total += invalids_uncovered[i];
    }

    cout << "Found invalids: " << invalids_total << endl;

    clauses_with_lit = (int*)malloc(feature_count * sizeof(int));
    current_reference_iterators = (int*)malloc(feature_count * sizeof(int));
    reference_start_points = (unsigned long long*)malloc(feature_count * sizeof(unsigned long long));

    for (int i = 0; i < feature_count; i++) clauses_with_lit[i] = 0;

    for (int i = 0; i < reduced_clause_count * lits_in_clause_limit; i++) {
        if (clause_list[i] != 0) {
            lit_value = abs(clause_list[i]) - 1;
            clauses_with_lit[lit_value]++;
        }
    }

    int max_references_to_clauses = 0;
    unsigned long long cumulated_references_to_clauses = 0;

    for (int i = 0; i < feature_count; i++) {
        if (clauses_with_lit[i] > max_references_to_clauses) {
            max_references_to_clauses = clauses_with_lit[i];
        }

        reference_start_points[i] = cumulated_references_to_clauses;
        current_reference_iterators[i] = 0;

        cumulated_references_to_clauses += clauses_with_lit[i];
    }

    cout << "\nMax clauses: " << max_references_to_clauses << endl;
    cout << "Total lits in file: " << total_lits_in_file << endl;
    cout << "Mean clauses: " << float(total_lits_in_file) / float(feature_count) << endl;

    int clause_index, lit_index_i, lit_index_j;
    unsigned long long inter_index;

    feat_to_clause_reference = (int*)malloc(cumulated_references_to_clauses * sizeof(int));
    clause_to_feat_reference = (int*)malloc(cumulated_references_to_clauses * sizeof(int));
    are_feature_pairs_dependent = (char*)malloc(feature_interaction_count * sizeof(char));

    for (int i = 0; i < reduced_clause_count * lits_in_clause_limit; i++) {
        if (clause_list[i] != 0) {
            clause_index = (i - i % lits_in_clause_limit) / lits_in_clause_limit + 1;
            lit_value = abs(clause_list[i]) - 1;

            if (clause_list[i] > 0) feat_to_clause_reference[reference_start_points[lit_value] + current_reference_iterators[lit_value]] = clause_index;
            else feat_to_clause_reference[reference_start_points[lit_value] + current_reference_iterators[lit_value]] = -clause_index;
            current_reference_iterators[lit_value]++;
        }
    }

    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < clauses_with_lit[i]; j++) {
            clause_to_feat_reference[reference_start_points[i] + j] = i;
        }
    }

    for (int i = 1; i < feature_count; i++) {
        for (int j = 0; j < i; j++) {
            inter_index = ((i - 1) * i) / 2 + j;
            are_feature_pairs_dependent[inter_index] = 0;
        }
    }

    for (int i = 0; i < reduced_clause_count; i++) {

        for (int j = 0; j < clause_list_sizes[i]; j++) {

            lit_index_i = clause_list[lits_in_clause_limit * i + j];
            lit_index_i = abs(lit_index_i) - 1;

            for (int k = 0; k < clause_list_sizes[i]; k++) {

                if (j != k) {
                    lit_index_j = clause_list[lits_in_clause_limit * i + k];
                    lit_index_j = abs(lit_index_j) - 1;

                    if (lit_index_i > lit_index_j) {
                        inter_index = ((lit_index_i - 1) * lit_index_i) / 2 + lit_index_j;
                        are_feature_pairs_dependent[inter_index] = 1;
                    }
                }
            }
        }
    }

    unsigned long long dependent_counter = 0;

    for (int i = 1; i < feature_count; i++) {
        for (int j = 0; j < i; j++) {
            inter_index = ((i - 1) * i) / 2 + j;
            if (are_feature_pairs_dependent[inter_index] == 1) dependent_counter++;
        }
    }

    cout << "Dependent interactions / total interactions: " << dependent_counter << " / " << feature_interaction_count << endl;

    auto elapsed0 = std::chrono::high_resolution_clock::now() - start0;

    long long microseconds0 = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed0).count();

    cout << "\nCNF file processing duration (us, ms, s): " << microseconds0 << ", " << microseconds0 / 1000 << ", " << microseconds0 / 1000000 << endl;

    cout << "\n=======================================\n" << endl;

    const int max_sample_size = 50000;

    if (sampled_variants_size > feature_count - 1) {
        cout << "ERROR! Sample size must be smaller than feature count. Exiting..." << endl;
        exit(0);
    }

    const int post_optimization_samples = 10;
    const bool disable_qip = false;
    const int optimization_epochs = 1000;
    const int max_retries_per_optimization = 25;
    const int min_retries_per_optimization = 7;
    int retries_per_optimization = max_retries_per_optimization;
    const bool dual_flip = false;
    const int step_size = 10;
    const bool modify_oldest = true;
    const bool limit_optimizations_to_sample_size = true;
    const int blockSize = 256;
    const int parallel_tasks = 100000;

    int numBlocksFeatures = int(round(static_cast<double>(feature_count + blockSize - 1) / static_cast<double>(blockSize)));
    int numBlocksClauses = int(round(static_cast<double>(reduced_clause_count + blockSize - 1) / static_cast<double>(blockSize)));

    int* current_sample_size, * current_sample_size_c;
    char* current_sample, * current_sample_c;
    int* skip_sample_index, * skip_sample_index_c;
    char* sampled_variants, * sampled_variants_c;
    unsigned long long* novel_fi_counts, * novel_fi_counts_c;
    unsigned long long* novel_fi_count, * novel_fi_count_c;
    char* feature_interactions, * feature_interactions_c;
    current_sample_size = (int*)malloc(sizeof(int));
    current_sample = (char*)malloc(feature_count * max_sample_size * sizeof(char));
    skip_sample_index = (int*)malloc(sizeof(int));
    sampled_variants = (char*)malloc(feature_count * sampled_variants_size * sizeof(char));
    novel_fi_counts = (unsigned long long*)malloc(sampled_variants_size * sizeof(unsigned long long));
    novel_fi_count = (unsigned long long*)malloc(sizeof(unsigned long long));
    feature_interactions = (char*)malloc(feature_interaction_count * sizeof(char));

    int* one_counts_c;

    int* max_index, * max_index_c;
    max_index = (int*)malloc(sizeof(int));

    int* max_index_m, * max_index_m_c;
    max_index_m = (int*)malloc(sizeof(int));

    int* max_gain_m;
    max_gain_m = (int*)malloc(sizeof(int));

    int* max_indices_j_m, * max_indices_j_m_c;
    max_indices_j_m = (int*)malloc(feature_count * sizeof(int));

    int* max_gains_m, * max_gains_m_c;
    max_gains_m = (int*)malloc(feature_count * sizeof(int));

    int* max_index_i_m, * max_index_i_m_c;
    max_index_i_m = (int*)malloc(sizeof(int));

    int* max_index_j_m, * max_index_j_m_c;
    max_index_j_m = (int*)malloc(sizeof(int));

    int* current_step_m, * current_step_m_c;
    current_step_m = (int*)malloc(sizeof(int));

    int* M_values, * M_values_c;
    M_values = (int*)malloc(feature_count * sizeof(int));

    char* optimized_variant, * optimized_variant_temp, * optimized_variant_c;
    optimized_variant = (char*)malloc(feature_count * sizeof(char));
    optimized_variant_temp = (char*)malloc(feature_count * sizeof(char));

    int* uncovered, * uncovered_c;
    uncovered = (int*)malloc((feature_count - 1) * sizeof(int));

    int* variant_ages;
    variant_ages = (int*)malloc(max_sample_size * sizeof(int));

    int* is_flip_valid, * is_flip_valid_c;
    is_flip_valid = (int*)malloc(feature_count * sizeof(int));

    int* shuffled_features;
    shuffled_features = (int*)malloc((feature_count - 1) * sizeof(int));

    unsigned long long* invalids_total_list;
    invalids_total_list = (unsigned long long*)malloc((sampled_variants_size) * sizeof(unsigned long long));

    cudaMalloc(&clause_list_c, init_clause_count * lits_in_clause_limit * sizeof(int));
    cudaMalloc(&clause_list_sizes_c, init_clause_count * sizeof(short));
    cudaMalloc(&current_clause_values_c, reduced_clause_count * sizeof(short));
    cudaMalloc(&min_clause_values_c, reduced_clause_count * sizeof(short));

    cudaMalloc(&clauses_with_lit_c, feature_count * sizeof(int));
    cudaMalloc(&reference_start_points_c, feature_count * sizeof(unsigned long long));
    cudaMalloc(&feat_to_clause_reference_c, cumulated_references_to_clauses * sizeof(int));
    cudaMalloc(&clause_to_feat_reference_c, cumulated_references_to_clauses * sizeof(int));

    cudaMalloc(&is_flip_valid_c, feature_count * sizeof(int));
    cudaMalloc(&dual_invalids_c, feature_interaction_count * sizeof(int));
    cudaMalloc(&dual_valids_c, feature_interaction_count * sizeof(int));
    cudaMalloc(&valid_thresholds_c, feature_count * sizeof(int));
    cudaMalloc(&are_feature_pairs_dependent_c, feature_interaction_count * sizeof(char));

    cudaMalloc(&current_sample_size_c, sizeof(int));
    cudaMalloc(&current_sample_c, feature_count * max_sample_size * sizeof(char));
    cudaMalloc(&skip_sample_index_c, sizeof(int));
    cudaMalloc(&sampled_variants_c, feature_count * sampled_variants_size * sizeof(char));
    cudaMalloc(&novel_fi_counts_c, sampled_variants_size * sizeof(unsigned long long));
    cudaMalloc(&novel_fi_count_c, sizeof(unsigned long long));
    cudaMalloc(&feature_interactions_c, feature_interaction_count * sizeof(char));
    cudaMalloc(&one_counts_c, feature_count * sizeof(int));
    cudaMalloc(&max_index_c, sizeof(int));
    cudaMalloc(&max_index_m_c, sizeof(int));
    cudaMalloc(&current_step_m_c, sizeof(int));

    cudaMalloc(&max_indices_j_m_c, feature_count * sizeof(int));
    cudaMalloc(&max_gains_m_c, feature_count * sizeof(int));

    cudaMalloc(&max_index_i_m_c, sizeof(int));
    cudaMalloc(&max_index_j_m_c, sizeof(int));

    cudaMalloc(&M_values_c, feature_count * sizeof(int));
    cudaMalloc(&optimized_variant_c, feature_count * sizeof(char));
    cudaMalloc(&uncovered_c, (feature_count - 1) * sizeof(int));

    curandState* rand_state;
    cudaMalloc(&rand_state, feature_count * sizeof(curandState));

    char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
    char inter_bin_encoding;
    char higher_value, lower_value;

    double rand_num;
    current_sample_size[0] = 1;

    for (int i = 0; i < feature_count * max_sample_size; i++) {
        if (i < feature_count * current_sample_size[0]) {
            rand_num = unif(rng);
            if (rand_num > 0.5) current_sample[i] = 1;
            else current_sample[i] = 0;
        }
        else current_sample[i] = 0;
    }

    S[0].resetSolver(feature_count);
    S[0].resetSimpSolver(feature_count);

    S[0].setTargetValue(0, feature_count, current_sample);

    S[0].eliminate(true);

    if (!S[0].okay()) {
        printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
        exit(0);
    }

    vec<Lit> dummy0;
    lbool ret0 = S[0].solveLimited(dummy0);

    if (ret0 == l_True) {
        for (int i = 0; i < feature_count; i++) {
            if (S[0].model[i] != l_Undef) {
                if (S[0].model[i] == l_True) current_sample[i] = 1;
                else current_sample[i] = 0;
            }
            else {
                printf("UNDEFINED VARIABLES, exiting...\n");
                exit(0);
            }
        }
    }
    else {
        printf("UNSATISFIABLE, exiting...\n");
        exit(0);
    }

    int age_counter = 1;
    variant_ages[0] = age_counter;
    age_counter++;

    skip_sample_index[0] = -1;

    std::thread myThreads[sampled_variants_size];

    for (int i = 0; i < feature_count - 1; i++) {
        shuffled_features[i] = i;
    }

    auto start1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(clause_list_c, clause_list, init_clause_count * lits_in_clause_limit * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clause_list_sizes_c, clause_list_sizes, init_clause_count * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(clauses_with_lit_c, clauses_with_lit, feature_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(reference_start_points_c, reference_start_points, feature_count * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(feat_to_clause_reference_c, feat_to_clause_reference, cumulated_references_to_clauses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clause_to_feat_reference_c, clause_to_feat_reference, cumulated_references_to_clauses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(are_feature_pairs_dependent_c, are_feature_pairs_dependent, feature_interaction_count * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(current_sample_size_c, current_sample_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(current_sample_c, current_sample, feature_count * max_sample_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(skip_sample_index_c, skip_sample_index, sizeof(int), cudaMemcpyHostToDevice);

    reset_uncovered << < numBlocksFeatures, blockSize >> > (feature_count, uncovered_c);
    calculate_interactions << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, current_sample_size_c, current_sample_c, skip_sample_index_c, feature_interactions_c, uncovered_c); // Threads: max. 1024

    vector<unsigned long long> uncovered_fi;
    vector<unsigned long long> invalid_fi;
    vector<int> variant_id;

    int variant_counter = 1;

    for (int n = 2; n < max_sample_size - 1; n++) {
        const int a = int(round(unif2(rng)));
        const int b = int(round(unif2(rng)));

        init_rand << < numBlocksFeatures, blockSize >> > (rand_state, a, b, feature_count);

        cudaMemcpy(uncovered, uncovered_c, (feature_count - 1) * sizeof(int), cudaMemcpyDeviceToHost);

        unsigned long long uncovered_total = 0;

        for (int i = 0; i < feature_count - 1; i++)  uncovered_total += uncovered[i];

        cout << ">>>>>> Uncovered interactions: " << uncovered_total << "  Invalid interactions: " << invalids_total << " <<<<<< " << endl;

        uncovered_fi.push_back(uncovered_total);
        invalid_fi.push_back(invalids_total);
        variant_id.push_back(variant_counter);
        variant_counter++;

        if (uncovered_total == invalids_total) {
            cout << "===============================" << endl;
            cout << "End reached...\n" << endl;
            break;
        }

        calculate_feature_probability << < numBlocksFeatures, blockSize >> > (feature_count, current_sample_size_c, current_sample_c, one_counts_c);
        init_sample << < numBlocksFeatures, blockSize >> > (rand_state, feature_count, current_sample_size_c, sampled_variants_size, sampled_variants_c, one_counts_c);

        cudaMemcpy(sampled_variants, sampled_variants_c, feature_count * sampled_variants_size * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(feature_interactions, feature_interactions_c, feature_interaction_count * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(feature_interactions, feature_interactions_c, feature_interaction_count * sizeof(char), cudaMemcpyDeviceToHost);

        random_shuffle(&shuffled_features[0], &shuffled_features[feature_count - 1]);

        cout << "Solver:" << endl;

        int start_point_x = 0, start_point_j = 0;

        bool found_valid, skip, end_reached = false;
        unsigned long long inter_index;
        char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
        char invalid_inter_off_off, invalid_inter_off_on, invalid_inter_on_off, invalid_inter_on_on;
        char inter_bin_encoding, invalid_inter_bin_encoding;

        for (int m = 0; m < sampled_variants_size; m++) {

            skip = false;

            found_valid = false;

            if (!end_reached) {
                if (start_point_x != 0 || start_point_j != 0) skip = true;

                for (int x = start_point_x; x < feature_count - 1; x++) {

                    int i = shuffled_features[x];

                    if (invalids_uncovered[i] < uncovered[i]) {
                        for (int j = start_point_j; j < i + 1; j++) {

                            if (!skip) {
                                inter_index = (i * (i + 1)) / 2 + j;

                                inter_bin_encoding = feature_interactions[inter_index];
                                invalid_inter_bin_encoding = invalid_feature_interactions[inter_index];

                                if (inter_bin_encoding != 0 && inter_bin_encoding != invalid_inter_bin_encoding) {

                                    if (inter_bin_encoding >= 8) {
                                        inter_on_on = 1;
                                        inter_bin_encoding -= 8;
                                    }
                                    else {
                                        inter_on_on = 0;
                                    }

                                    if (inter_bin_encoding >= 4) {
                                        inter_on_off = 1;
                                        inter_bin_encoding -= 4;
                                    }
                                    else {
                                        inter_on_off = 0;
                                    }

                                    if (inter_bin_encoding >= 2) {
                                        inter_off_on = 1;
                                        inter_bin_encoding -= 2;
                                    }
                                    else {
                                        inter_off_on = 0;
                                    }

                                    if (inter_bin_encoding == 1) inter_off_off = 1;
                                    else inter_off_off = 0;

                                    if (invalid_inter_bin_encoding >= 8) {
                                        invalid_inter_on_on = 1;
                                        invalid_inter_bin_encoding -= 8;
                                    }
                                    else {
                                        invalid_inter_on_on = 0;
                                    }

                                    if (invalid_inter_bin_encoding >= 4) {
                                        invalid_inter_on_off = 1;
                                        invalid_inter_bin_encoding -= 4;
                                    }
                                    else {
                                        invalid_inter_on_off = 0;
                                    }

                                    if (invalid_inter_bin_encoding >= 2) {
                                        invalid_inter_off_on = 1;
                                        invalid_inter_bin_encoding -= 2;
                                    }
                                    else {
                                        invalid_inter_off_on = 0;
                                    }

                                    if (invalid_inter_bin_encoding == 1) invalid_inter_off_off = 1;
                                    else invalid_inter_off_off = 0;

                                    if (invalid_inter_on_on == 0 && inter_on_on == 1) {
                                        S[0].resetSolver(feature_count);
                                        S[0].resetSimpSolver(feature_count);

                                        S[0].setTargetValue(feature_count * m, feature_count, sampled_variants);

                                        S[0].eliminate(true);

                                        if (!S[0].okay()) {
                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                            exit(0);
                                        }

                                        vec<Lit> dummy1;

                                        dummy1.push(mkLit(i + 1));
                                        dummy1.push(mkLit(j));

                                        lbool ret1 = S[0].solveLimited(dummy1);

                                        if (ret1 == l_True) {
                                            for (int k = 0; k < feature_count; k++) {
                                                if (S[0].model[k] != l_Undef) {
                                                    if (S[0].model[k] == l_True) sampled_variants[m * feature_count + k] = 1;
                                                    else sampled_variants[m * feature_count + k] = 0;
                                                }
                                                else {
                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                    exit(0);
                                                }
                                            }

                                            start_point_x = x;
                                            start_point_j = j;

                                            found_valid = true;
                                            break;
                                        }
                                        else {
                                            invalid_inter_on_on = 1;

                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                            invalids_uncovered[i]++;

                                            invalids_total++;

                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                        }
                                    }

                                    if (invalid_inter_on_off == 0 && inter_on_off == 1) {
                                        S[0].resetSolver(feature_count);
                                        S[0].resetSimpSolver(feature_count);

                                        S[0].setTargetValue(feature_count * m, feature_count, sampled_variants);

                                        S[0].eliminate(true);

                                        if (!S[0].okay()) {
                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                            exit(0);
                                        }

                                        vec<Lit> dummy1;

                                        dummy1.push(mkLit(i + 1));
                                        dummy1.push(~mkLit(j));

                                        lbool ret1 = S[0].solveLimited(dummy1);
                                        if (ret1 == l_True) {
                                            for (int k = 0; k < feature_count; k++) {
                                                if (S[0].model[k] != l_Undef) {
                                                    if (S[0].model[k] == l_True) sampled_variants[m * feature_count + k] = 1;
                                                    else sampled_variants[m * feature_count + k] = 0;
                                                }
                                                else {
                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                    exit(0);
                                                }
                                            }

                                            start_point_x = x;
                                            start_point_j = j;

                                            found_valid = true;
                                            break;
                                        }
                                        else {
                                            invalid_inter_on_off = 1;

                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                            invalids_uncovered[i]++;

                                            invalids_total++;

                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                        }
                                    }

                                    if (invalid_inter_off_on == 0 && inter_off_on == 1) {
                                        S[0].resetSolver(feature_count);
                                        S[0].resetSimpSolver(feature_count);

                                        S[0].setTargetValue(feature_count * m, feature_count, sampled_variants);

                                        S[0].eliminate(true);

                                        if (!S[0].okay()) {
                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                            exit(0);
                                        }

                                        vec<Lit> dummy1;

                                        dummy1.push(~mkLit(i + 1));
                                        dummy1.push(mkLit(j));

                                        lbool ret1 = S[0].solveLimited(dummy1);
                                        if (ret1 == l_True) {
                                            for (int k = 0; k < feature_count; k++) {
                                                if (S[0].model[k] != l_Undef) {
                                                    if (S[0].model[k] == l_True) sampled_variants[m * feature_count + k] = 1;
                                                    else sampled_variants[m * feature_count + k] = 0;
                                                }
                                                else {
                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                    exit(0);
                                                }
                                            }

                                            start_point_x = x;
                                            start_point_j = j;

                                            found_valid = true;
                                            break;
                                        }
                                        else {
                                            invalid_inter_off_on = 1;

                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                            invalids_uncovered[i]++;

                                            invalids_total++;

                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                        }
                                    }

                                    if (invalid_inter_off_off == 0 && inter_off_off == 1) {
                                        S[0].resetSolver(feature_count);
                                        S[0].resetSimpSolver(feature_count);

                                        S[0].setTargetValue(feature_count * m, feature_count, sampled_variants);

                                        S[0].eliminate(true);

                                        if (!S[0].okay()) {
                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                            exit(0);
                                        }

                                        vec<Lit> dummy1;

                                        dummy1.push(~mkLit(i + 1));
                                        dummy1.push(~mkLit(j));

                                        lbool ret1 = S[0].solveLimited(dummy1);
                                        if (ret1 == l_True) {
                                            for (int k = 0; k < feature_count; k++) {
                                                if (S[0].model[k] != l_Undef) {
                                                    if (S[0].model[k] == l_True) sampled_variants[m * feature_count + k] = 1;
                                                    else sampled_variants[m * feature_count + k] = 0;
                                                }
                                                else {
                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                    exit(0);
                                                }
                                            }

                                            start_point_x = x;
                                            start_point_j = j;

                                            found_valid = true;
                                            break;
                                        }
                                        else {
                                            invalid_inter_off_off = 1;

                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                            invalids_uncovered[i]++;

                                            invalids_total++;

                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                        }
                                    }

                                }
                            }
                            else skip = false;
                        }
                    }
                    if (found_valid) break;
                }
            }

            if (!found_valid) {
                end_reached = true;

                S[0].resetSolver(feature_count);
                S[0].resetSimpSolver(feature_count);

                S[0].setTargetValue(feature_count * m, feature_count, sampled_variants);

                S[0].eliminate(true);

                if (!S[0].okay()) {
                    printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                    exit(0);
                }

                vec<Lit> dummy1;
                lbool ret1 = S[0].solveLimited(dummy1);

                if (ret1 == l_True) {

                    for (int k = 0; k < feature_count; k++) {
                        if (S[0].model[k] != l_Undef) {
                            if (S[0].model[k] == l_True) sampled_variants[m * feature_count + k] = 1;
                            else sampled_variants[m * feature_count + k] = 0;
                        }
                        else {
                            printf("UNDEFINED VARIABLES, exiting...\n");
                            exit(0);
                        }
                    }
                }
                else {
                    printf("UNSAT, exiting...\n");
                    exit(0);
                }
            }
        }

        if (uncovered_total == invalids_total) {
            cout << "===============================" << endl;
            cout << "End reached...\n" << endl;
            break;
        }

        cudaMemcpy(sampled_variants_c, sampled_variants, feature_count * sampled_variants_size * sizeof(char), cudaMemcpyHostToDevice);

        reset_novel_fi_counts << < 1, 1 >> > (sampled_variants_size, novel_fi_counts_c);
        calculate_sample_gain_all << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, sampled_variants_size, sampled_variants_c, novel_fi_counts_c, feature_interactions_c);

        find_max_sample << < 1, 1 >> > (sampled_variants_size, novel_fi_counts_c, max_index_c);
        write_max_sample_to_optimized << < numBlocksFeatures, blockSize >> > (feature_count, sampled_variants_c, optimized_variant_c, max_index_c);

        reset_M_values << < numBlocksFeatures, blockSize >> > (feature_count, M_values_c);
        calculate_init_M_values << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interactions_c, optimized_variant_c, M_values_c);

        calculate_clause_values << < numBlocksClauses, blockSize >> > (reduced_clause_count, current_clause_values_c, min_clause_values_c, clause_list_c, lits_in_clause_limit, clause_list_sizes_c, optimized_variant_c);

        int it_count = 0;
        int zero_counter = 0;
        unsigned long long total_gain = 0;

        while (it_count < optimization_epochs && !disable_qip) {

            if (!dual_flip) {

                reset_flip_counts << < numBlocksFeatures, blockSize >> > (feature_count, is_flip_valid_c);

                calculate_flip_validity_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, total_lits_in_file, current_clause_values_c, min_clause_values_c, feat_to_clause_reference_c, clause_to_feat_reference_c, optimized_variant_c, is_flip_valid_c);

                cudaMemcpy(M_values, M_values_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(optimized_variant, optimized_variant_c, feature_count * sizeof(char), cudaMemcpyDeviceToHost);
                cudaMemcpy(is_flip_valid, is_flip_valid_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);

                max_index_m[0] = -1;
                max_gain_m[0] = 0;

                int gain_value;
                char value;

                for (int i = 0; i < feature_count; i++) {
                    if (is_flip_valid[i] == 0) {
                        value = optimized_variant[i];
                        gain_value = M_values[i] * (1 - 2 * value);

                        if (gain_value > max_gain_m[0]) {
                            max_gain_m[0] = gain_value;
                            max_index_m[0] = i;
                        }
                    }
                }

                if (max_gain_m[0] > 0) {
                    cudaMemcpy(max_index_m_c, max_index_m, sizeof(int), cudaMemcpyHostToDevice);

                    adapt_M_values << < numBlocksFeatures, blockSize >> > (feature_count, feature_interactions_c, optimized_variant_c, M_values_c, max_index_m_c);

                    int numBlocksAffectedClauses = int(round(static_cast<double>(clauses_with_lit[max_index_m[0]]) + blockSize - 1) / static_cast<double>(blockSize));

                    adapt_clause_values << < numBlocksAffectedClauses, blockSize >> > (current_clause_values_c, clauses_with_lit_c, feat_to_clause_reference_c, reference_start_points_c, optimized_variant_c, max_index_m_c);

                    optimize_variant << < 1, 1 >> > (optimized_variant_c, max_index_m_c);
                    total_gain += max_gain_m[0];
                }
                else break;

            }
            else {
                current_step_m[0] = it_count % step_size;

                cudaMemcpy(current_step_m_c, current_step_m, sizeof(int), cudaMemcpyHostToDevice);

                reset_dual_flip_counts << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, is_flip_valid_c, dual_invalids_c, dual_valids_c, valid_thresholds_c);

                calculate_dual_flip_validity_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, total_lits_in_file, current_clause_values_c, min_clause_values_c, feat_to_clause_reference_c, clause_to_feat_reference_c, clause_list_c, clause_list_sizes_c, lits_in_clause_limit, optimized_variant_c, is_flip_valid_c, dual_invalids_c, dual_valids_c, valid_thresholds_c);

                calculate_max_gain_dual_with_validity_check_2 << < numBlocksFeatures, blockSize >> > (feature_count, feature_interactions_c, optimized_variant_c, M_values_c, max_indices_j_m_c, max_gains_m_c, step_size, current_step_m_c, is_flip_valid_c, dual_invalids_c, dual_valids_c, valid_thresholds_c, are_feature_pairs_dependent_c);

                cudaMemcpy(max_gains_m, max_gains_m_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(max_indices_j_m, max_indices_j_m_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);
                
                int gain_value = 0;

                for (int i = 0; i < feature_count; i++) {
                    if (max_gains_m[i] > gain_value) {
                        gain_value = max_gains_m[i];
                        max_index_i_m[0] = i;
                        max_index_j_m[0] = max_indices_j_m[i];
                    }
                }

                if (gain_value > 0) {
                    cudaMemcpy(max_index_i_m_c, max_index_i_m, sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(max_index_j_m_c, max_index_j_m, sizeof(int), cudaMemcpyHostToDevice);

                    adapt_M_values_dual << < numBlocksFeatures, blockSize >> > (feature_count, feature_interactions_c, optimized_variant_c, M_values_c, max_index_i_m_c, max_index_j_m_c);

                    int numBlocksAffectedClauses = int(round(static_cast<double>(clauses_with_lit[max_index_i_m[0]]) + blockSize - 1) / static_cast<double>(blockSize));

                    adapt_clause_values << < numBlocksAffectedClauses, blockSize >> > (current_clause_values_c, clauses_with_lit_c, feat_to_clause_reference_c, reference_start_points_c, optimized_variant_c, max_index_i_m_c);

                    if (max_index_j_m[0] != -1) {
                        numBlocksAffectedClauses = int(round(static_cast<double>(clauses_with_lit[max_index_j_m[0]]) + blockSize - 1) / static_cast<double>(blockSize));
                        adapt_clause_values << < numBlocksAffectedClauses, blockSize >> > (current_clause_values_c, clauses_with_lit_c, feat_to_clause_reference_c, reference_start_points_c, optimized_variant_c, max_index_j_m_c);
                    }

                    optimize_variant_dual << < 1, 1 >> > (optimized_variant_c, max_index_i_m_c, max_index_j_m_c);

                    total_gain += gain_value;

                    zero_counter = 0;
                }
                else {
                    zero_counter++;

                    if (zero_counter == step_size) break;
                }
            }

            it_count++;
        }

        /*
        cudaMemcpy(optimized_variant, optimized_variant_c, feature_count * sizeof(char), cudaMemcpyDeviceToHost);

        ifstream file1(cnf_file_name);

        bool found;
        int sat_counter = 0;
        int unsat_counter = 0;
        int number;
        bool negative, comment, command;

        if (file1.is_open()) {
            while (getline(file1, file_line)) {
                number = 0;
                negative = false;
                found = false;

                comment = false;
                command = false;

                char_counter = 0;

                for (char& c : file_line) {
                    int index = int(c);

                    if (char_counter == 0 && index == 99) {
                        comment = true;
                        break;
                    }

                    if (char_counter == 0 && index == 112) {
                        command = true;
                        break;
                    }

                    if (index >= 48 && index <= 57) {
                        if (number == 0) {
                            number = index - 48;
                        }
                        else {
                            number *= 10;
                            number += index - 48;
                        }
                    }
                    else if (index == 45) {
                        negative = true;
                    }
                    else {
                        if (number > 0) {
                            if (negative) {
                                if (optimized_variant[number - 1] == 0) found = true;
                            }
                            else {
                                if (optimized_variant[number - 1] == 1) found = true;
                            }
                        }

                        number = 0;
                        negative = false;
                    }

                    char_counter++;
                }

                if (comment || command) continue;

                if (number > 0) {
                    if (negative) {
                        if (optimized_variant[number - 1] == 0) found = true;
                    }
                    else {
                        if (optimized_variant[number - 1] == 1) found = true;
                    }
                }

                if (!found) unsat_counter++;
                else sat_counter++;
            }

            file1.close();
        }

        std::cout << "\nCLAUSES SAT / UNSAT: " << sat_counter << " / " << unsat_counter << std::endl;
        */

        cudaMemcpy(novel_fi_counts, novel_fi_counts_c, sampled_variants_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        cudaMemcpy(max_index, max_index_c, sizeof(int), cudaMemcpyDeviceToHost);

        write_optimized_to_current_sample << < numBlocksFeatures, blockSize >> > (feature_count, optimized_variant_c, current_sample_size_c, current_sample_c);

        if (post_optimization_samples == 0) append_interactions << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, optimized_variant_c, feature_interactions_c, uncovered_c);

        increase_current_sample_size << < 1, 1 >> > (current_sample_size_c);

        if (modify_oldest) {
            variant_ages[n - 1] = age_counter;
            age_counter++;
        }

        cout << ">>>>>> Variant " << n << " <<<<<<" << endl;

        cout << "New covered interactions before: " << novel_fi_counts[max_index[0]] << " - Optimized: " << novel_fi_counts[max_index[0]] + total_gain << " Factor: " << float(novel_fi_counts[max_index[0]] + total_gain) / float(novel_fi_counts[max_index[0]]) << endl;

        cout << "-------------------------------" << endl;

        uncovered_total -= novel_fi_counts[max_index[0]] + total_gain;

        if (post_optimization_samples > 0) {

            std::uniform_real_distribution<double> unif3(0, n - 1);

            int limit;
            int not_zero_counter = 0;

            if (limit_optimizations_to_sample_size) limit = std::min(post_optimization_samples, n);
            else limit = post_optimization_samples;

            for (int m = 0; m < limit; m++) {

                if (modify_oldest) {
                    int min_age = age_counter;

                    for (int i = 0; i < n; i++) {
                        if (variant_ages[i] < min_age) {
                            skip_sample_index[0] = i;
                            min_age = variant_ages[i];
                        }
                    }
                }
                else skip_sample_index[0] = int(round(unif3(rng)));

                cudaMemcpy(skip_sample_index_c, skip_sample_index, sizeof(int), cudaMemcpyHostToDevice);

                reset_uncovered << < numBlocksFeatures, blockSize >> > (feature_count, uncovered_c);
                calculate_interactions << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, current_sample_size_c, current_sample_c, skip_sample_index_c, feature_interactions_c, uncovered_c); // Threads: max. 1024

                cudaMemcpy(uncovered, uncovered_c, (feature_count - 1) * sizeof(int), cudaMemcpyDeviceToHost);

                unsigned long long uncovered_without = 0;

                for (int i = 0; i < feature_count - 1; i++) {
                    uncovered_without += uncovered[i];
                }

                write_selected_sample_to_optimized << < numBlocksFeatures, blockSize >> > (feature_count, current_sample_c, optimized_variant_c, skip_sample_index_c);

                reset_M_values << < numBlocksFeatures, blockSize >> > (feature_count, M_values_c);
                calculate_init_M_values << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interactions_c, optimized_variant_c, M_values_c);

                calculate_clause_values << < numBlocksClauses, blockSize >> > (reduced_clause_count, current_clause_values_c, min_clause_values_c, clause_list_c, lits_in_clause_limit, clause_list_sizes_c, optimized_variant_c);

                it_count = 0;

                total_gain = 0;

                zero_counter = 0;

                int run_counter = 0;

                int start_point_x = 0, start_point_j = 0;

                random_shuffle(&shuffled_features[0], &shuffled_features[feature_count - 1]);

                bool use_alternative = false;

                while (it_count < optimization_epochs) {

                    if (!dual_flip) {
                        
                        if (!disable_qip) {
                            reset_flip_counts << < numBlocksFeatures, blockSize >> > (feature_count, is_flip_valid_c);

                            calculate_flip_validity_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, total_lits_in_file, current_clause_values_c, min_clause_values_c, feat_to_clause_reference_c, clause_to_feat_reference_c, optimized_variant_c, is_flip_valid_c);

                            cudaMemcpy(M_values, M_values_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(optimized_variant, optimized_variant_c, feature_count * sizeof(char), cudaMemcpyDeviceToHost);
                            cudaMemcpy(is_flip_valid, is_flip_valid_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);

                            max_index_m[0] = -1;
                            max_gain_m[0] = 0;

                            int gain_value;
                            char value;

                            for (int i = 0; i < feature_count; i++) {
                                if (is_flip_valid[i] == 0) {
                                    value = optimized_variant[i];

                                    gain_value = M_values[i] * (1 - 2 * value);

                                    if (gain_value > max_gain_m[0]) {
                                        max_gain_m[0] = gain_value;
                                        max_index_m[0] = i;
                                    }
                                }
                            }
                        } else max_gain_m[0] = 0;

                        if (max_gain_m[0] > 0) {
                            cudaMemcpy(max_index_m_c, max_index_m, sizeof(int), cudaMemcpyHostToDevice);

                            adapt_M_values << < numBlocksFeatures, blockSize >> > (feature_count, feature_interactions_c, optimized_variant_c, M_values_c, max_index_m_c);

                            int numBlocksAffectedClauses = int(round(static_cast<double>(clauses_with_lit[max_index_m[0]]) + blockSize - 1) / static_cast<double>(blockSize));

                            adapt_clause_values << < numBlocksAffectedClauses, blockSize >> > (current_clause_values_c, clauses_with_lit_c, feat_to_clause_reference_c, reference_start_points_c, optimized_variant_c, max_index_m_c);

                            optimize_variant << < 1, 1 >> > (optimized_variant_c, max_index_m_c);
                            
                            if (run_counter == 0) total_gain += max_gain_m[0];
                        }
                        else {
                            if (total_gain > 0) break;

                            if (run_counter == 0) {
                                cudaMemcpy(optimized_variant, optimized_variant_c, feature_count * sizeof(char), cudaMemcpyDeviceToHost);
                                cudaMemcpy(feature_interactions, feature_interactions_c, feature_interaction_count * sizeof(char), cudaMemcpyDeviceToHost);

                                for (int i = 0; i < feature_count; i++) optimized_variant_temp[i] = optimized_variant[i];
                            }

                            if (run_counter < retries_per_optimization) {

                                if (run_counter >= 1) {
                                    reset_novel_fi_count << < 1, 1 >> > (novel_fi_count_c);
                                    calculate_sample_gain_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, optimized_variant_c, novel_fi_count_c, feature_interactions_c);
                                    cudaMemcpy(novel_fi_count, novel_fi_count_c, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

                                    if (novel_fi_count[0] > uncovered_without - uncovered_total + total_gain) {
                                        cout << "Found something better!" << endl;
                                        use_alternative = true;
                                        break;
                                    }
                                }

                                run_counter++;

                                bool skip = false;

                                if (start_point_x != 0 || start_point_j != 0) skip = true;

                                bool found_valid = false;
                                unsigned long long inter_index;
                                char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
                                char invalid_inter_off_off, invalid_inter_off_on, invalid_inter_on_off, invalid_inter_on_on;
                                char inter_bin_encoding, invalid_inter_bin_encoding;

                                for (int x = start_point_x; x < feature_count - 1; x++) {

                                    int i = shuffled_features[x];

                                    if (invalids_uncovered[i] < uncovered[i]) {
                                        for (int j = start_point_j; j < i + 1; j++) {

                                            if (!skip) {
                                                inter_index = (i * (i + 1)) / 2 + j;

                                                inter_bin_encoding = feature_interactions[inter_index];
                                                invalid_inter_bin_encoding = invalid_feature_interactions[inter_index];

                                                if (inter_bin_encoding != 0 && inter_bin_encoding != invalid_inter_bin_encoding) {

                                                    if (inter_bin_encoding >= 8) {
                                                        inter_on_on = 1;
                                                        inter_bin_encoding -= 8;
                                                    }
                                                    else {
                                                        inter_on_on = 0;
                                                    }

                                                    if (inter_bin_encoding >= 4) {
                                                        inter_on_off = 1;
                                                        inter_bin_encoding -= 4;
                                                    }
                                                    else {
                                                        inter_on_off = 0;
                                                    }

                                                    if (inter_bin_encoding >= 2) {
                                                        inter_off_on = 1;
                                                        inter_bin_encoding -= 2;
                                                    }
                                                    else {
                                                        inter_off_on = 0;
                                                    }

                                                    if (inter_bin_encoding == 1) inter_off_off = 1;
                                                    else inter_off_off = 0;

                                                    if (invalid_inter_bin_encoding >= 8) {
                                                        invalid_inter_on_on = 1;
                                                        invalid_inter_bin_encoding -= 8;
                                                    }
                                                    else {
                                                        invalid_inter_on_on = 0;
                                                    }

                                                    if (invalid_inter_bin_encoding >= 4) {
                                                        invalid_inter_on_off = 1;
                                                        invalid_inter_bin_encoding -= 4;
                                                    }
                                                    else {
                                                        invalid_inter_on_off = 0;
                                                    }

                                                    if (invalid_inter_bin_encoding >= 2) {
                                                        invalid_inter_off_on = 1;
                                                        invalid_inter_bin_encoding -= 2;
                                                    }
                                                    else {
                                                        invalid_inter_off_on = 0;
                                                    }

                                                    if (invalid_inter_bin_encoding == 1) invalid_inter_off_off = 1;
                                                    else invalid_inter_off_off = 0;

                                                    if (invalid_inter_on_on == 0 && inter_on_on == 1 && !(optimized_variant_temp[i + 1] == 1 && optimized_variant_temp[j] == 1)) {
                                                        S[0].resetSolver(feature_count);
                                                        S[0].resetSimpSolver(feature_count);

                                                        S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                        S[0].eliminate(true);

                                                        if (!S[0].okay()) {
                                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                            exit(0);
                                                        }

                                                        vec<Lit> dummy1;

                                                        dummy1.push(mkLit(i + 1));
                                                        dummy1.push(mkLit(j));

                                                        lbool ret1 = S[0].solveLimited(dummy1);

                                                        if (ret1 == l_True) {
                                                            for (int k = 0; k < feature_count; k++) {
                                                                if (S[0].model[k] != l_Undef) {
                                                                    if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                    else optimized_variant[k] = 0;
                                                                }
                                                                else {
                                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                                    exit(0);
                                                                }
                                                            }

                                                            start_point_x = x;
                                                            start_point_j = j;

                                                            found_valid = true;
                                                            break;
                                                        }
                                                        else {
                                                            invalid_inter_on_on = 1;

                                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                            invalids_uncovered[i]++;

                                                            invalids_total++;

                                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                                        }
                                                    }

                                                    if (invalid_inter_on_off == 0 && inter_on_off == 1 && !(optimized_variant_temp[i + 1] == 1 && optimized_variant_temp[j] == 0)) {
                                                        S[0].resetSolver(feature_count);
                                                        S[0].resetSimpSolver(feature_count);

                                                        S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                        S[0].eliminate(true);

                                                        if (!S[0].okay()) {
                                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                            exit(0);
                                                        }

                                                        vec<Lit> dummy1;

                                                        dummy1.push(mkLit(i + 1));
                                                        dummy1.push(~mkLit(j));

                                                        lbool ret1 = S[0].solveLimited(dummy1);
                                                        if (ret1 == l_True) {
                                                            for (int k = 0; k < feature_count; k++) {
                                                                if (S[0].model[k] != l_Undef) {
                                                                    if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                    else optimized_variant[k] = 0;
                                                                }
                                                                else {
                                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                                    exit(0);
                                                                }
                                                            }

                                                            start_point_x = x;
                                                            start_point_j = j;

                                                            found_valid = true;
                                                            break;
                                                        }
                                                        else {
                                                            invalid_inter_on_off = 1;

                                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                            invalids_uncovered[i]++;

                                                            invalids_total++;

                                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                                        }
                                                    }

                                                    if (invalid_inter_off_on == 0 && inter_off_on == 1 && !(optimized_variant_temp[i + 1] == 0 && optimized_variant_temp[j] == 1)) {
                                                        S[0].resetSolver(feature_count);
                                                        S[0].resetSimpSolver(feature_count);

                                                        S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                        S[0].eliminate(true);

                                                        if (!S[0].okay()) {
                                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                            exit(0);
                                                        }

                                                        vec<Lit> dummy1;

                                                        dummy1.push(~mkLit(i + 1));
                                                        dummy1.push(mkLit(j));

                                                        lbool ret1 = S[0].solveLimited(dummy1);
                                                        if (ret1 == l_True) {
                                                            for (int k = 0; k < feature_count; k++) {
                                                                if (S[0].model[k] != l_Undef) {
                                                                    if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                    else optimized_variant[k] = 0;
                                                                }
                                                                else {
                                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                                    exit(0);
                                                                }
                                                            }

                                                            start_point_x = x;
                                                            start_point_j = j;

                                                            found_valid = true;
                                                            break;
                                                        }
                                                        else {
                                                            invalid_inter_off_on = 1;

                                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                            invalids_uncovered[i]++;

                                                            invalids_total++;

                                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                                        }
                                                    }

                                                    if (invalid_inter_off_off == 0 && inter_off_off == 1 && !(optimized_variant_temp[i + 1] == 0 && optimized_variant_temp[j] == 0)) {
                                                        S[0].resetSolver(feature_count);
                                                        S[0].resetSimpSolver(feature_count);

                                                        S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                        S[0].eliminate(true);

                                                        if (!S[0].okay()) {
                                                            printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                            exit(0);
                                                        }

                                                        vec<Lit> dummy1;

                                                        dummy1.push(~mkLit(i + 1));
                                                        dummy1.push(~mkLit(j));

                                                        lbool ret1 = S[0].solveLimited(dummy1);
                                                        if (ret1 == l_True) {
                                                            for (int k = 0; k < feature_count; k++) {
                                                                if (S[0].model[k] != l_Undef) {
                                                                    if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                    else optimized_variant[k] = 0;
                                                                }
                                                                else {
                                                                    printf("UNDEFINED VARIABLES, exiting...\n");
                                                                    exit(0);
                                                                }
                                                            }

                                                            start_point_x = x;
                                                            start_point_j = j;

                                                            found_valid = true;
                                                            break;
                                                        }
                                                        else {
                                                            invalid_inter_off_off = 1;

                                                            invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                            invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                            invalids_uncovered[i]++;

                                                            invalids_total++;

                                                            if (invalids_uncovered[i] == uncovered[i]) break;
                                                        }
                                                    }

                                                }
                                            }
                                            else skip = false;
                                        }
                                    }
                                    if (found_valid) break;
                                }

                                if (!found_valid) {
                                    cout << "End reached, exiting..." << endl;
                                    break;
                                }

                                cudaMemcpy(optimized_variant_c, optimized_variant, feature_count * sizeof(char), cudaMemcpyHostToDevice);

                                reset_M_values << < numBlocksFeatures, blockSize >> > (feature_count, M_values_c);
                                calculate_init_M_values << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interactions_c, optimized_variant_c, M_values_c);

                                calculate_clause_values << < numBlocksClauses, blockSize >> > (reduced_clause_count, current_clause_values_c, min_clause_values_c, clause_list_c, lits_in_clause_limit, clause_list_sizes_c, optimized_variant_c);
                            }
                            else break;

                        }

                    }
                    else {
                        int gain_value;

                        if (!disable_qip) {
                            current_step_m[0] = it_count % step_size;

                            cudaMemcpy(current_step_m_c, current_step_m, sizeof(int), cudaMemcpyHostToDevice);

                            reset_dual_flip_counts << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, is_flip_valid_c, dual_invalids_c, dual_valids_c, valid_thresholds_c);

                            calculate_dual_flip_validity_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, total_lits_in_file, current_clause_values_c, min_clause_values_c, feat_to_clause_reference_c, clause_to_feat_reference_c, clause_list_c, clause_list_sizes_c, lits_in_clause_limit, optimized_variant_c, is_flip_valid_c, dual_invalids_c, dual_valids_c, valid_thresholds_c);

                            calculate_max_gain_dual_with_validity_check_2 << < numBlocksFeatures, blockSize >> > (feature_count, feature_interactions_c, optimized_variant_c, M_values_c, max_indices_j_m_c, max_gains_m_c, step_size, current_step_m_c, is_flip_valid_c, dual_invalids_c, dual_valids_c, valid_thresholds_c, are_feature_pairs_dependent_c);

                            cudaMemcpy(max_gains_m, max_gains_m_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(max_indices_j_m, max_indices_j_m_c, feature_count * sizeof(int), cudaMemcpyDeviceToHost);

                            gain_value = 0;

                            for (int i = 0; i < feature_count; i++) {
                                if (max_gains_m[i] > gain_value) {
                                    gain_value = max_gains_m[i];
                                    max_index_i_m[0] = i;
                                    max_index_j_m[0] = max_indices_j_m[i];
                                }
                            }
                        }
                        else gain_value = 0;

                        if (gain_value > 0) {
                            cudaMemcpy(max_index_i_m_c, max_index_i_m, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(max_index_j_m_c, max_index_j_m, sizeof(int), cudaMemcpyHostToDevice);

                            adapt_M_values_dual << < numBlocksFeatures, blockSize >> > (feature_count, feature_interactions_c, optimized_variant_c, M_values_c, max_index_i_m_c, max_index_j_m_c);

                            int numBlocksAffectedClauses = int(round(static_cast<double>(clauses_with_lit[max_index_i_m[0]]) + blockSize - 1) / static_cast<double>(blockSize));

                            adapt_clause_values << < numBlocksAffectedClauses, blockSize >> > (current_clause_values_c, clauses_with_lit_c, feat_to_clause_reference_c, reference_start_points_c, optimized_variant_c, max_index_i_m_c);

                            if (max_index_j_m[0] != -1) {
                                numBlocksAffectedClauses = int(round(static_cast<double>(clauses_with_lit[max_index_j_m[0]]) + blockSize - 1) / static_cast<double>(blockSize));
                                adapt_clause_values << < numBlocksAffectedClauses, blockSize >> > (current_clause_values_c, clauses_with_lit_c, feat_to_clause_reference_c, reference_start_points_c, optimized_variant_c, max_index_j_m_c);
                            }

                            optimize_variant_dual << < 1, 1 >> > (optimized_variant_c, max_index_i_m_c, max_index_j_m_c);

                            if (run_counter == 0) total_gain += gain_value;

                            zero_counter = 0;
                        }
                        else {
                            zero_counter++;

                            if (zero_counter == step_size) {

                                if (total_gain > 0) break;

                                if (run_counter == 0) {
                                    cudaMemcpy(optimized_variant, optimized_variant_c, feature_count * sizeof(char), cudaMemcpyDeviceToHost);
                                    cudaMemcpy(feature_interactions, feature_interactions_c, feature_interaction_count * sizeof(char), cudaMemcpyDeviceToHost);

                                    for (int i = 0; i < feature_count; i++) optimized_variant_temp[i] = optimized_variant[i];
                                }

                                if (run_counter < retries_per_optimization) {

                                    if (run_counter >= 1) {
                                        reset_novel_fi_count << < 1, 1 >> > (novel_fi_count_c);
                                        calculate_sample_gain_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, optimized_variant_c, novel_fi_count_c, feature_interactions_c);
                                        cudaMemcpy(novel_fi_count, novel_fi_count_c, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

                                        if (novel_fi_count[0] > uncovered_without - uncovered_total + total_gain) {
                                            cout << "Found something better!" << endl;
                                            use_alternative = true;
                                            break;
                                        }
                                    }

                                    zero_counter = 0;
                                    run_counter++;

                                    bool skip = false;

                                    if (start_point_x != 0 || start_point_j != 0) skip = true;

                                    bool found_valid = false;
                                    unsigned long long inter_index;
                                    char inter_off_off, inter_off_on, inter_on_off, inter_on_on;
                                    char invalid_inter_off_off, invalid_inter_off_on, invalid_inter_on_off, invalid_inter_on_on;
                                    char inter_bin_encoding, invalid_inter_bin_encoding;

                                    for (int x = start_point_x; x < feature_count - 1; x++) {

                                        int i = shuffled_features[x];

                                        if (invalids_uncovered[i] < uncovered[i]) {
                                            for (int j = start_point_j; j < i + 1; j++) {

                                                if (!skip) {
                                                    inter_index = (i * (i + 1)) / 2 + j;

                                                    inter_bin_encoding = feature_interactions[inter_index];
                                                    invalid_inter_bin_encoding = invalid_feature_interactions[inter_index];

                                                    if (inter_bin_encoding != 0 && inter_bin_encoding != invalid_inter_bin_encoding) {

                                                        if (inter_bin_encoding >= 8) {
                                                            inter_on_on = 1;
                                                            inter_bin_encoding -= 8;
                                                        }
                                                        else {
                                                            inter_on_on = 0;
                                                        }

                                                        if (inter_bin_encoding >= 4) {
                                                            inter_on_off = 1;
                                                            inter_bin_encoding -= 4;
                                                        }
                                                        else {
                                                            inter_on_off = 0;
                                                        }

                                                        if (inter_bin_encoding >= 2) {
                                                            inter_off_on = 1;
                                                            inter_bin_encoding -= 2;
                                                        }
                                                        else {
                                                            inter_off_on = 0;
                                                        }

                                                        if (inter_bin_encoding == 1) inter_off_off = 1;
                                                        else inter_off_off = 0;

                                                        if (invalid_inter_bin_encoding >= 8) {
                                                            invalid_inter_on_on = 1;
                                                            invalid_inter_bin_encoding -= 8;
                                                        }
                                                        else {
                                                            invalid_inter_on_on = 0;
                                                        }

                                                        if (invalid_inter_bin_encoding >= 4) {
                                                            invalid_inter_on_off = 1;
                                                            invalid_inter_bin_encoding -= 4;
                                                        }
                                                        else {
                                                            invalid_inter_on_off = 0;
                                                        }

                                                        if (invalid_inter_bin_encoding >= 2) {
                                                            invalid_inter_off_on = 1;
                                                            invalid_inter_bin_encoding -= 2;
                                                        }
                                                        else {
                                                            invalid_inter_off_on = 0;
                                                        }

                                                        if (invalid_inter_bin_encoding == 1) invalid_inter_off_off = 1;
                                                        else invalid_inter_off_off = 0;

                                                        if (invalid_inter_on_on == 0 && inter_on_on == 1 && !(optimized_variant_temp[i + 1] == 1 && optimized_variant_temp[j] == 1)) {
                                                            S[0].resetSolver(feature_count);
                                                            S[0].resetSimpSolver(feature_count);

                                                            S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                            S[0].eliminate(true);

                                                            if (!S[0].okay()) {
                                                                printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                                exit(0);
                                                            }

                                                            vec<Lit> dummy1;

                                                            dummy1.push(mkLit(i + 1));
                                                            dummy1.push(mkLit(j));

                                                            lbool ret1 = S[0].solveLimited(dummy1);

                                                            if (ret1 == l_True) {
                                                                for (int k = 0; k < feature_count; k++) {
                                                                    if (S[0].model[k] != l_Undef) {
                                                                        if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                        else optimized_variant[k] = 0;
                                                                    }
                                                                    else {
                                                                        printf("UNDEFINED VARIABLES, exiting...\n");
                                                                        exit(0);
                                                                    }
                                                                }

                                                                start_point_x = x;
                                                                start_point_j = j;

                                                                found_valid = true;
                                                                break;
                                                            }
                                                            else {
                                                                invalid_inter_on_on = 1;

                                                                invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                                invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                                invalids_uncovered[i]++;

                                                                invalids_total++;

                                                                if (invalids_uncovered[i] == uncovered[i]) break;
                                                            }
                                                        }

                                                        if (invalid_inter_on_off == 0 && inter_on_off == 1 && !(optimized_variant_temp[i + 1] == 1 && optimized_variant_temp[j] == 0)) {
                                                            S[0].resetSolver(feature_count);
                                                            S[0].resetSimpSolver(feature_count);

                                                            S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                            S[0].eliminate(true);

                                                            if (!S[0].okay()) {
                                                                printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                                exit(0);
                                                            }

                                                            vec<Lit> dummy1;

                                                            dummy1.push(mkLit(i + 1));
                                                            dummy1.push(~mkLit(j));

                                                            lbool ret1 = S[0].solveLimited(dummy1);
                                                            if (ret1 == l_True) {
                                                                for (int k = 0; k < feature_count; k++) {
                                                                    if (S[0].model[k] != l_Undef) {
                                                                        if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                        else optimized_variant[k] = 0;
                                                                    }
                                                                    else {
                                                                        printf("UNDEFINED VARIABLES, exiting...\n");
                                                                        exit(0);
                                                                    }
                                                                }

                                                                start_point_x = x;
                                                                start_point_j = j;

                                                                found_valid = true;
                                                                break;
                                                            }
                                                            else {
                                                                invalid_inter_on_off = 1;

                                                                invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                                invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                                invalids_uncovered[i]++;

                                                                invalids_total++;

                                                                if (invalids_uncovered[i] == uncovered[i]) break;
                                                            }
                                                        }

                                                        if (invalid_inter_off_on == 0 && inter_off_on == 1 && !(optimized_variant_temp[i + 1] == 0 && optimized_variant_temp[j] == 1)) {
                                                            S[0].resetSolver(feature_count);
                                                            S[0].resetSimpSolver(feature_count);

                                                            S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                            S[0].eliminate(true);

                                                            if (!S[0].okay()) {
                                                                printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                                exit(0);
                                                            }

                                                            vec<Lit> dummy1;

                                                            dummy1.push(~mkLit(i + 1));
                                                            dummy1.push(mkLit(j));

                                                            lbool ret1 = S[0].solveLimited(dummy1);
                                                            if (ret1 == l_True) {
                                                                for (int k = 0; k < feature_count; k++) {
                                                                    if (S[0].model[k] != l_Undef) {
                                                                        if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                        else optimized_variant[k] = 0;
                                                                    }
                                                                    else {
                                                                        printf("UNDEFINED VARIABLES, exiting...\n");
                                                                        exit(0);
                                                                    }
                                                                }

                                                                start_point_x = x;
                                                                start_point_j = j;

                                                                found_valid = true;
                                                                break;
                                                            }
                                                            else {
                                                                invalid_inter_off_on = 1;

                                                                invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                                invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                                invalids_uncovered[i]++;

                                                                invalids_total++;

                                                                if (invalids_uncovered[i] == uncovered[i]) break;
                                                            }
                                                        }

                                                        if (invalid_inter_off_off == 0 && inter_off_off == 1 && !(optimized_variant_temp[i + 1] == 0 && optimized_variant_temp[j] == 0)) {
                                                            S[0].resetSolver(feature_count);
                                                            S[0].resetSimpSolver(feature_count);

                                                            S[0].setTargetValue(0, feature_count, optimized_variant_temp);

                                                            S[0].eliminate(true);

                                                            if (!S[0].okay()) {
                                                                printf("Solved by simplification - UNSATISFIABLE, exiting...\n");
                                                                exit(0);
                                                            }

                                                            vec<Lit> dummy1;

                                                            dummy1.push(~mkLit(i + 1));
                                                            dummy1.push(~mkLit(j));

                                                            lbool ret1 = S[0].solveLimited(dummy1);
                                                            if (ret1 == l_True) {
                                                                for (int k = 0; k < feature_count; k++) {
                                                                    if (S[0].model[k] != l_Undef) {
                                                                        if (S[0].model[k] == l_True) optimized_variant[k] = 1;
                                                                        else optimized_variant[k] = 0;
                                                                    }
                                                                    else {
                                                                        printf("UNDEFINED VARIABLES, exiting...\n");
                                                                        exit(0);
                                                                    }
                                                                }

                                                                start_point_x = x;
                                                                start_point_j = j;

                                                                found_valid = true;
                                                                break;
                                                            }
                                                            else {
                                                                invalid_inter_off_off = 1;

                                                                invalid_inter_bin_encoding = invalid_inter_off_off + 2 * invalid_inter_off_on + 4 * invalid_inter_on_off + 8 * invalid_inter_on_on;

                                                                invalid_feature_interactions[inter_index] = invalid_inter_bin_encoding;

                                                                invalids_uncovered[i]++;

                                                                invalids_total++;

                                                                if (invalids_uncovered[i] == uncovered[i]) break;
                                                            }
                                                        }

                                                    }
                                                }
                                                else skip = false;
                                            }
                                        }
                                        if (found_valid) break;
                                    }

                                    if (!found_valid) {
                                        cout << "End reached, exiting..." << endl;
                                        break;
                                    }

                                    cudaMemcpy(optimized_variant_c, optimized_variant, feature_count * sizeof(char), cudaMemcpyHostToDevice);

                                    reset_M_values << < numBlocksFeatures, blockSize >> > (feature_count, M_values_c);
                                    calculate_init_M_values << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interactions_c, optimized_variant_c, M_values_c);

                                    calculate_clause_values << < numBlocksClauses, blockSize >> > (reduced_clause_count, current_clause_values_c, min_clause_values_c, clause_list_c, lits_in_clause_limit, clause_list_sizes_c, optimized_variant_c);
                                }
                                else break;
                            }
                        }

                    }

                    it_count++;
                }

                if (!use_alternative) {
                    if (run_counter >= 1) {
                        reset_novel_fi_count << < 1, 1 >> > (novel_fi_count_c);
                        calculate_sample_gain_optimized << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, optimized_variant_c, novel_fi_count_c, feature_interactions_c);
                        cudaMemcpy(novel_fi_count, novel_fi_count_c, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

                        if (novel_fi_count[0] > uncovered_without - uncovered_total + total_gain) {
                            cout << "Found something better!" << endl;
                            use_alternative = true;
                        }
                    }
                }

                unsigned uncovered_total_before = uncovered_total;

                if (!use_alternative) {
                    if (total_gain == 0) {
                        for (int i = 0; i < feature_count; i++) optimized_variant[i] = optimized_variant_temp[i];
                        cudaMemcpy(optimized_variant_c, optimized_variant, feature_count * sizeof(char), cudaMemcpyHostToDevice);
                    }

                    uncovered_total -= total_gain;
                }
                else {
                    uncovered_total = uncovered_without - novel_fi_count[0];
                }

                if (uncovered_total_before > uncovered_total) {
                    not_zero_counter++;
                }

                /*
                if (uncovered_total_before > uncovered_total) {
                    cudaMemcpy(optimized_variant, optimized_variant_c, feature_count * sizeof(char), cudaMemcpyDeviceToHost);

                    ifstream file1(cnf_file_name);

                    bool found;
                    int sat_counter = 0;
                    int unsat_counter = 0;
                    int number;
                    bool negative, comment, command;

                    if (file1.is_open()) {
                        while (getline(file1, file_line)) {
                            number = 0;
                            negative = false;
                            found = false;

                            comment = false;
                            command = false;

                            char_counter = 0;

                            for (char& c : file_line) {
                                int index = int(c);

                                if (char_counter == 0 && index == 99) {
                                    comment = true;
                                    break;
                                }

                                if (char_counter == 0 && index == 112) {
                                    command = true;
                                    break;
                                }

                                if (index >= 48 && index <= 57) {
                                    if (number == 0) {
                                        number = index - 48;
                                    }
                                    else {
                                        number *= 10;
                                        number += index - 48;
                                    }
                                }
                                else if (index == 45) {
                                    negative = true;
                                }
                                else {
                                    if (number > 0) {
                                        if (negative) {
                                            if (optimized_variant[number - 1] == 0) found = true;
                                        }
                                        else {
                                            if (optimized_variant[number - 1] == 1) found = true;
                                        }
                                    }

                                    number = 0;
                                    negative = false;
                                }

                                char_counter++;
                            }

                            if (comment || command) continue;

                            if (number > 0) {
                                if (negative) {
                                    if (optimized_variant[number - 1] == 0) found = true;
                                }
                                else {
                                    if (optimized_variant[number - 1] == 1) found = true;
                                }
                            }

                            if (!found) unsat_counter++;
                            else sat_counter++;
                        }

                        file1.close();
                    }

                    std::cout << "\nCLAUSES SAT / UNSAT: " << sat_counter << " / " << unsat_counter << std::endl;
                    if (unsat_counter != 0) cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                }
                */

                write_optimized_to_current_sample << < numBlocksFeatures, blockSize >> > (feature_count, optimized_variant_c, skip_sample_index_c, current_sample_c);

                if (modify_oldest) {
                    variant_ages[skip_sample_index[0]] = age_counter;
                    age_counter++;
                }

                append_interactions << < int(round(static_cast<double>(parallel_tasks) + blockSize - 1) / static_cast<double>(blockSize)), blockSize >> > (parallel_tasks, feature_count, feature_interaction_count, optimized_variant_c, feature_interactions_c, uncovered_c);

                cout << "Variant: " << skip_sample_index[0] << " - newly covered interactions: " << uncovered_total_before - uncovered_total << " - iterations: " << it_count + 1 << endl;
            }


            if (not_zero_counter > 2) retries_per_optimization++;
            else retries_per_optimization--;

            if (retries_per_optimization > max_retries_per_optimization) retries_per_optimization = max_retries_per_optimization;
            if (retries_per_optimization < min_retries_per_optimization) retries_per_optimization = min_retries_per_optimization;

        }

        cout << "===============================" << endl;

    }

    cudaMemcpy(uncovered, uncovered_c, (feature_count - 1) * sizeof(int), cudaMemcpyDeviceToHost);

    unsigned long long uncovered_total = 0;

    for (int i = 0; i < feature_count - 1; i++) {
        uncovered_total += uncovered[i];
    }

    cout << ">>>>>> Uncovered interactions: " << uncovered_total << " <<<<<<" << endl;

    uncovered_fi.push_back(uncovered_total);
    invalid_fi.push_back(invalids_total);
    variant_id.push_back(variant_counter);

    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;

    long long microseconds1 = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed1).count();

    cout << "On GPU duration (us, ms, s): " << microseconds1 << ", " << microseconds1 / 1000 << ", " << microseconds1 / 1000000 << endl;

    cout << "=======================================" << endl;


    for (int i = 0; i < uncovered_fi.size(); i++) {
        cout << uncovered_fi[i] << ",";
    }
    cout << endl;

    for (int i = 0; i < uncovered_fi.size(); i++) {
        cout << invalid_fi[i] << ",";
    }
    cout << endl;

    for (int i = 0; i < uncovered_fi.size(); i++) {
        cout << variant_id[i] << ",";
    }
    cout << endl;

    /*
    char invalid_inter_bin_encoding;
    char invalid_inter_off_off, invalid_inter_off_on, invalid_inter_on_off, invalid_inter_on_on;

    std::ofstream file("C:\\Users\\lenna\\Downloads\\invalid_interactions\\erp_system.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }

    std::vector<std::vector<std::string>> csv_data;


    for (int i = 1; i < feature_count; i++) {

        for (int j = 0; j < i; j++) {

            inter_index = ((i - 1) * i) / 2 + j;

            invalid_inter_bin_encoding = invalid_feature_interactions[inter_index];

            if (inter_bin_encoding == 1) inter_off_off = 1;
            else inter_off_off = 0;

            if (invalid_inter_bin_encoding >= 8) {
                invalid_inter_on_on = 1;
                invalid_inter_bin_encoding -= 8;
            }
            else {
                invalid_inter_on_on = 0;
            }

            if (invalid_inter_bin_encoding >= 4) {
                invalid_inter_on_off = 1;
                invalid_inter_bin_encoding -= 4;
            }
            else {
                invalid_inter_on_off = 0;
            }

            if (invalid_inter_bin_encoding >= 2) {
                invalid_inter_off_on = 1;
                invalid_inter_bin_encoding -= 2;
            }
            else {
                invalid_inter_off_on = 0;
            }

            if (invalid_inter_bin_encoding == 1) invalid_inter_off_off = 1;
            else invalid_inter_off_off = 0;

            if (invalid_inter_on_on == 1) csv_data.push_back({ "" + std::to_string(i+1), "" + std::to_string(j+1)});
            if (invalid_inter_on_off == 1) csv_data.push_back({ "" + std::to_string(i + 1), "-" + std::to_string(j + 1) });
            if (invalid_inter_off_on == 1) csv_data.push_back({ "-" + std::to_string(i + 1), "" + std::to_string(j + 1) });
            if (invalid_inter_off_off == 1) csv_data.push_back({ "-" + std::to_string(i + 1), "-" + std::to_string(j + 1) });
        }
    }

    for (const auto& row : csv_data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i != row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "CSV file created successfully." << std::endl;
    */

    cudaFree(clause_list_c);
    cudaFree(clause_list_sizes_c);
    cudaFree(current_clause_values_c);
    cudaFree(min_clause_values_c);
    cudaFree(feat_to_clause_reference_c);
    cudaFree(clauses_with_lit_c);
    cudaFree(reference_start_points_c);
    cudaFree(is_flip_valid_c);

    cudaFree(current_sample_size_c);
    cudaFree(current_sample_c);
    cudaFree(skip_sample_index_c);
    cudaFree(sampled_variants_c);
    cudaFree(novel_fi_counts_c);
    cudaFree(novel_fi_count_c);
    cudaFree(feature_interactions_c);
    cudaFree(one_counts_c);
    cudaFree(max_index_c);
    cudaFree(max_index_m_c);

    cudaFree(max_indices_j_m_c);
    cudaFree(max_gains_m_c);
    cudaFree(max_index_i_m_c);
    cudaFree(max_index_j_m_c);
    cudaFree(current_step_m_c);

    cudaFree(M_values_c);
    cudaFree(optimized_variant_c);
    cudaFree(uncovered_c);

    cout << "\nCUDA cleanup completed..." << endl;

    free(invalid_feature_interactions);
    free(invalids_uncovered);
    free(active_literals);
    free(inactive_literals);
    free(clause_list);
    free(clause_list_sizes);
    free(feat_to_clause_reference);
    free(clauses_with_lit);
    free(current_reference_iterators);
    free(reference_start_points);
    free(is_flip_valid);

    free(current_sample_size);
    free(current_sample);
    free(skip_sample_index);
    free(sampled_variants);
    free(novel_fi_counts);
    free(feature_interactions);
    free(max_index);
    free(max_index_m);
    free(max_gain_m);

    free(max_indices_j_m);
    free(max_gains_m);
    free(max_index_i_m);
    free(max_index_j_m);
    free(current_step_m);

    free(M_values);
    free(optimized_variant);
    free(optimized_variant_temp);
    free(uncovered);

    cout << "Host cleanup completed..." << endl;
}