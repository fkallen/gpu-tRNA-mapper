#include "parsing.cuh"

#include "kseqpp/kseqpp.hpp"

#include "common.cuh"

#include <string>

#include <nvtx3/nvtx3.hpp>

std::string_view sequenceHeaderToNameView(std::string_view header){
    auto pos = header.find_first_of(' ');
    if(pos == std::string::npos){
        return header;
    }else{
        return header.substr(0, pos);
    }
}

int parse_n_sequences(kseqpp::KseqPP& reader, SequenceCollection& output, int N, bool withHeaders, bool withQualityScores){
    NVTX3_FUNC_RANGE();

    int parsedSequences = 0;

    while(reader.next() > 0){
        const auto& seq = reader.getCurrentSequence();
        output.sumOfLengths += seq.size();
        output.h_lengths.push_back(seq.size());
        output.h_offsets.push_back(output.h_sequences.size());
        output.h_sequences.insert(output.h_sequences.end(), seq.begin(), seq.end());
        const size_t sequencepadding = (seq.size() % 4 == 0) ? 0 : 4 - seq.size() % 4;
        output.h_sequences.insert(output.h_sequences.end(), sequencepadding, ' ');
        output.maximumSequenceLength = std::max(output.maximumSequenceLength , int(seq.size()));

        if(withQualityScores){
            const auto& qual = reader.getCurrentQuality();
            output.qualities.insert(output.qualities.end(), qual.begin(), qual.end());
            output.qualities.insert(output.qualities.end(), sequencepadding, ' ');
        }

        if(withHeaders){
            output.headers.push_back(std::move(reader.getCurrentHeader()));
            output.sequenceNames.emplace_back(sequenceHeaderToNameView(output.headers.back()));
        }
        parsedSequences++;
        if(parsedSequences == N){
            break;
        }
    }

    return parsedSequences;
}

SequenceCollection parseAllSequences(const std::string& filename, bool withHeaders, bool withQualityScores){
    SequenceCollection collection;
    collection.hasQualityScores = withQualityScores;
    kseqpp::KseqPP reader(filename);
    const int batchsize = 1000;

    while(batchsize == parse_n_sequences(reader, collection, batchsize, withHeaders, withQualityScores)){
        ;
    }
    
    return collection;
}
