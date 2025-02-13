#ifndef PARSING_HPP
#define PARSING_HPP

#include "kseqpp/kseqpp.hpp"

#include "common.cuh"

#include <string>



int parse_n_sequences(kseqpp::KseqPP& reader, SequenceCollection& output, int N, bool withHeaders, bool withQualityScores);

SequenceCollection parseAllSequences(const std::string& filename, bool withHeaders, bool withQualityScores);





#endif