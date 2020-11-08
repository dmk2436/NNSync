#include "nns.hpp"
#include "mkldnn.hpp"
void initialize() {
engine = mkldnn::engine(mkldnn::engine::cpu, 0);
}