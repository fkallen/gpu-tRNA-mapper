#ifndef SAM_HELPER_HPP
#define SAM_HELPER_HPP

#include <iostream>
#include <ostream>
#include <string>
#include <string_view>

#include "parasail_helpers.hpp"

struct SAM{
    static constexpr unsigned int SAMFLAGS_PAIRED(){ return 0x1; }
    static constexpr unsigned int SAMFLAGS_PROPER_PAIR(){ return 0x2; }
    static constexpr unsigned int SAMFLAGS_UNMAP(){ return 0x4; }
    static constexpr unsigned int SAMFLAGS_MUNMAP(){ return 0x8; }
    static constexpr unsigned int SAMFLAGS_REVERSE(){ return 0x10; }
    static constexpr unsigned int SAMFLAGS_MREVERSE(){ return 0x20; }
    static constexpr unsigned int SAMFLAGS_READ1(){ return 0x40; }
    static constexpr unsigned int SAMFLAGS_READ2(){ return 0x80; }
    static constexpr unsigned int SAMFLAGS_SECONDARY(){ return 0x100; }
    static constexpr unsigned int SAMFLAGS_QCFAIL(){ return 0x200; }
    static constexpr unsigned int SAMFLAGS_DUP(){ return 0x400; }
    static constexpr unsigned int SAMFLAGS_SUPPLEMENTARY(){ return 0x800; }
    static constexpr char SAM_SEPARATOR(){ return '\t'; }
};

template<class SequenceCollection>
void writeSAMheader(std::ostream& stream, const SequenceCollection* referenceSequences){
    stream << "@HD" << SAM::SAM_SEPARATOR() << "VN:1.4" << SAM::SAM_SEPARATOR() << "SO:queryname" << '\n';

    const int numReferenceSequences = referenceSequences->h_lengths.size();
    for(int i = 0; i < numReferenceSequences; i++){
        stream << "@SQ" << SAM::SAM_SEPARATOR() << "SN:" << referenceSequences->sequenceNames[i] << SAM::SAM_SEPARATOR() << "LN:" << referenceSequences->h_lengths[i] << '\n';
    }

    stream << "@PG" << SAM::SAM_SEPARATOR() << "ID:42" << SAM::SAM_SEPARATOR() << "PN:custom gpu aligner" << '\n';
}


//create the SAM line and append it to stream (inclusive '\n' at the end)
void writeAlignmentResultInSAMformat_SW(
    std::ostream& stream,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view referenceSequence,
    std::string_view readQuality,
    const ParasailResultData& alignmentResultWithTraceback,
    const parasail_matrix_t* scoringmatrix
);

void appendAlignmentResultInSAMformat_SW(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view referenceSequence,
    std::string_view readQuality,
    const ParasailResultData& alignmentResultWithTraceback,
    const parasail_matrix_t* scoringmatrix
);

void writeAlignmentResultInSAMformat_NW_as_SW(
    std::ostream& stream,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view referenceSequence,
    std::string_view readQuality,
    int SWreadBeginPos,
    int SWreadEndPos_excl,
    int SWrefBeginPos,
    int SWrefEndPos_excl,
    const ParasailResultData& NWresultWithTraceback,
    const parasail_matrix_t* scoringmatrix
);

void appendAlignmentResultInSAMformat_NW_to_SW(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view referenceSequence,
    std::string_view readQuality,
    int SWreadBeginPos,
    int SWreadEndPos_excl,
    int SWrefBeginPos,
    int SWrefEndPos_excl,
    const ParasailResultData& NWresultWithTraceback,
    const parasail_matrix_t* scoringmatrix
);


void appendAlignmentResultInSAMformat_SG(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view referenceSequence,
    std::string_view readQuality,
    const ParasailResultData& alignmentResultWithTraceback,
    const parasail_matrix_t* scoringmatrix
);

void appendAlignmentResultInSAMformat_NW_to_SG(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view referenceSequence,
    std::string_view readQuality,
    int SGreadBeginPos,
    int SGreadEndPos_excl,
    int SGrefBeginPos,
    int SGrefEndPos_excl,
    int score,
    const ParasailResultData& NWresultWithTraceback,
    const parasail_matrix_t* scoringmatrix
);



void writeUnmappedAlignmentResultInSAMformat(
    std::ostream& stream,
    std::string_view readName, 
    std::string_view readSequence,
    std::string_view readQuality
);



#endif