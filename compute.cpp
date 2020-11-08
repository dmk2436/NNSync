#include "compute.hpp"
#include <assert.h>
#include <chrono>
#include <array>
#include <vector>
#include <string>
#include <numeric>
#include "mkldnn.hpp"
void initialize() {
auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);
}
class layer {
private:
enum layer_type {
EMPTY, FCN, CONV, RELU, POOL, LRN
};
public:
std::vector<layer&> prev, next;
std::string name, desc;
std::array<uint, 3> input_dim, output_dim;
dnnl::memory dst_memory;
layer_type type;
layer() {
input_dim = prev[0].output_dim;
for(auto i: prev) {
if(input_dim != i.output_dim) {
// ERROR: padding mismatch
}
i.next.push_back((layer&)*this);
this->prev.push_back(i);
output_dim = input_dim;
}
}
virtual ~layer();

};
class fcn: layer {
fcn();
virtual ~fcn();
};
class conv: layer {
std::vector<float> conv_weights;
std::vector<float> conv_bias;
conv(std::vector<layer> &prev, int filters, int w, int h, int stride, int padding) {
output_dim = {(input_dim[0] - w) / stride + 1, (input_dim[1] - h) / stride + 1, filters};
mkldnn::memory::dims conv_src_tz = {batch, input_dim[2], input_dim[0], input_dim[1]};
mkldnn::memory::dims conv_weights_tz = {output_dim[2], input_dim[2], w, h};
mkldnn::memory::dims conv_bias_tz = {filters};
mkldnn::memory::dims conv_dst_tz = {batch, output_dim[2], output_dim[0], output_dim[1]};
mkldnn::memory::dims conv_strides = {stride, stride};
mkldnn::memory::dims conv_padding = {padding, padding};
auto conv_padding = {0, 0};
std::vector<float> conv_weights(std::accumulate(conv_weights_tz.begin(),
conv_weights_tz.end(), 1, std::multiplies<uint32_t>()));
std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(), conv_bias_tz.end(), 1,
std::multiplies<uint32_t>()));
auto conv_user_weights_memory = mkldnn::memory({{{conv_weights_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::oihw}, engine},
conv_weights.data());
auto conv_user_bias_memory = mkldnn::memory({{{conv_bias_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::x}, engine}, conv_bias.data());
auto conv_src_md = mkldnn::memory::desc({conv_src_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
auto conv_weights_md = mkldnn::memory::desc({conv_weights_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
auto conv_bias_md = mkldnn::memory::desc({conv_bias_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
auto conv_desc = mkldnn::convolution_forward::desc(mkldnn::pop_kind::forward,

mkldnn::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
conv_dst_md, conv_strides, conv_padding, conv_padding, mkldnn::padding_kind::zero);
auto conv_prim_desc = mkldnn::convolution_forward::primitive_desc(conv_desc, engine);
dst_memory = mkldnn::memory(conv_prim_desc.dst_primitive_desc());

// create memory for user data
auto conv5_user_weights_memory
= memory({{conv5_weights_tz}, dt::f32, tag::goihw}, eng);
write_to_dnnl_memory(conv5_weights.data(), conv5_user_weights_memory);
auto conv5_user_bias_memory
= memory({{conv5_bias_tz}, dt::f32, tag::x}, eng);
write_to_dnnl_memory(conv5_bias.data(), conv5_user_bias_memory);
// create memory descriptors for convolution data w/ no specified format
auto conv5_src_md = memory::desc({conv5_src_tz}, dt::f32, tag::any);
auto conv5_weights_md = memory::desc({conv5_weights_tz}, dt::f32, tag::any);
auto conv5_bias_md = memory::desc({conv5_bias_tz}, dt::f32, tag::any);
auto conv5_dst_md = memory::desc({conv5_dst_tz}, dt::f32, tag::any);
// create a convolution
auto conv5_desc = convolution_forward::desc(prop_kind::forward_inference,
algorithm::convolution_direct, conv5_src_md, conv5_weights_md,
conv5_bias_md, conv5_dst_md, conv5_strides, conv5_padding,
conv5_padding);
auto conv5_prim_desc = convolution_forward::primitive_desc(conv5_desc, eng);
auto conv5_src_memory = conv4_dst_memory;
if (conv5_prim_desc.src_desc() != conv5_src_memory.get_desc()) {
conv5_src_memory = memory(conv5_prim_desc.src_desc(), eng);
net.push_back(reorder(conv4_dst_memory, conv5_src_memory));
net_args.push_back({{DNNL_ARG_FROM, conv4_dst_memory},
{DNNL_ARG_TO, conv5_src_memory}});
}
auto conv5_weights_memory = conv5_user_weights_memory;
if (conv5_prim_desc.weights_desc()

!= conv5_user_weights_memory.get_desc()) {
conv5_weights_memory = memory(conv5_prim_desc.weights_desc(), eng);
reorder(conv5_user_weights_memory, conv5_weights_memory)
.execute(s, conv5_user_weights_memory, conv5_weights_memory);
}
auto conv5_dst_memory = memory(conv5_prim_desc.dst_desc(), eng);
}
virtual ~conv();
};
class relu: layer {
relu(std::vector<layer> &prev) {
const double negative_slope = 1.0;
dst_memory = mkldnn::memory(prev[0].dst_memory.dst_primitive_desc());
auto relu_desc = mkldnn::relu_forward::desc(mkldnn::prop_kind::forward,
conv_prim_desc.dst_primitive_desc().desc(), negative_slope);
}
virtual ~relu();
};
class pool: layer {
pool(std::vector<layer> &prev, int stride, padding) {
mkldnn::memory::dims pool_dst_tz = {batch, output_dim[2], output_dim[0], output_dim[1]};
mkldnn::memory::dims pool_kernel = {pool, pool};
mkldnn::memory::dims pool_strides = {stride, stride};
auto pool_padding = {0, 0};
auto pool_user_dst_memory = mkldnn::memory({{{pool_dst_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw}, engine}, net_dst.data());
auto pool_dst_md = mkldnn::memory::desc({pool_dst_tz},
mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
auto pool_desc = mkldnn::pooling_forward::desc(mkldnn::prop_kind::forward,
mkldnn::pooling_max, prev[0].dst_memory.get_primitive_desc().desc(), pool_dst_md,
pool_strides, pool_kernel, pool_padding, pool_padding, mkldnn::pading_kind::zero);
auto pool_pd = mkldnn::pooling_forward::primitive_desc(pool_desc, engine);
dst_memory = pool_user_dst_memory;

auto pool_indicies_memory = mkldnn::memory(dst_memory.get_primitive_desc());
}
virtual ~pool();
};
class lrn: layer {
lrn(std::vector<layer> &prev) {
const uint32_t local_size = 5;
const double alpha = 0.0001;
const double beta = 0.75;
dst_memory = mkldnn::memory(prev[0].dst_memory.get_primitive_desc());
auto lrn_scratch_memory = mkldnn::memory(dst_memory.get_primitive_desc());
auto lrn_desc = mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward,
mkldnn::lrn_across_channels, conv_prim_desc.dst_primitive_desc().desc(), local_size, alpha,
beta);
auto lrn_prim_desc = mkldnn::lrn_forward::primitive_desc(lrn_desc, engine);
}
virtual ~lrn();
};