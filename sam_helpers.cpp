#include "sam_helpers.hpp"

#include <iostream>
#include <ostream>
#include <string>
#include <string_view>

#include "parasail_helpers.hpp"




//create the SAM line and append it to stream (inclusive '\n' at the end)
void writeAlignmentResultInSAMformat_SW(
    std::ostream& stream,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view /*referenceSequence*/,
    std::string_view readQuality,
    const ParasailResultData& alignmentResultWithTraceback,
    const parasail_matrix_t* /*scoringmatrix*/
){

    parasail_result_t* parasail_result_ptr = alignmentResultWithTraceback.result.get();
    parasail_cigar_t* parasail_cigar_ptr = alignmentResultWithTraceback.cigar.get();

    // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
    //     parasail_result_get_cigar(
    //         alignmentResultWithTraceback,
    //         readSequence.data(), readSequence.size(),
    //         referenceSequence.data(), referenceSequence.size(),
    //         scoringmatrix
    //     ),
    //     parasail_cigar_free
    // );

    stream << readName << SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = 0;
    // if(secondary){
    //     flags |= SAM::SAMFLAGS_SECONDARY();
    // }
    stream << flags << SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    stream << referenceName << SAM::SAM_SEPARATOR();

    //POS (1-based, 0 == unmapped)
    stream << parasail_cigar_ptr->beg_ref + 1 << SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    stream << 255 << SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped)
    //soft clip at begin of read
    if (parasail_result_is_sw(parasail_result_ptr)) {
        if (parasail_cigar_ptr->beg_query > 0) {
            stream << parasail_cigar_ptr->beg_query << 'S';
        }
    }
    int editDistance = 0;

    for(int c=0; c < parasail_cigar_ptr->len; c++) {
        char letter = parasail_cigar_decode_op(parasail_cigar_ptr->seq[c]);
        uint32_t length = parasail_cigar_decode_len(parasail_cigar_ptr->seq[c]);
        stream << length << letter;
        if ('X' == letter || 'I' == letter || 'D' == letter) {
            editDistance += length;
        }
    }
    if (parasail_result_is_sw(parasail_result_ptr)) {
        const int remainderInRead = readSequence.size() - parasail_result_get_end_query(parasail_result_ptr) - 1;
        //soft clip at end of read
        if (remainderInRead > 0) {
            stream << remainderInRead << 'S';
        }
    }
    stream << SAM::SAM_SEPARATOR();
    // stream << parasailCigarString.get() << SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    stream << '*' << SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    stream << 0 << SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    stream << 0 << SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    stream << readSequence << SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        stream << '*' << SAM::SAM_SEPARATOR();
    }else{
        stream << readQuality << SAM::SAM_SEPARATOR();
    }

    // OPTIONAL FIELDS

    //score
    stream << "AS:i:" << parasail_result_get_score(parasail_result_ptr) << SAM::SAM_SEPARATOR();

    //edit distance
    stream << "NM:i:" << editDistance;

    stream << '\n';
}

void appendAlignmentResultInSAMformat_SW(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view /*referenceSequence*/,
    std::string_view readQuality,
    const ParasailResultData& alignmentResultWithTraceback,
    const parasail_matrix_t* /*scoringmatrix*/
){

    parasail_result_t* parasail_result_ptr = alignmentResultWithTraceback.result.get();
    parasail_cigar_t* parasail_cigar_ptr = alignmentResultWithTraceback.cigar.get();

    // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
    //     parasail_result_get_cigar(
    //         alignmentResultWithTraceback,
    //         readSequence.data(), readSequence.size(),
    //         referenceSequence.data(), referenceSequence.size(),
    //         scoringmatrix
    //     ),
    //     parasail_cigar_free
    // );

    sam_string += readName;
    sam_string += SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = 0;
    sam_string += std::to_string(flags);
    sam_string += SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    sam_string += referenceName;
    sam_string += SAM::SAM_SEPARATOR();

    //POS (1-based, 0 == unmapped)
    sam_string += std::to_string(parasail_cigar_ptr->beg_ref + 1);
    sam_string += SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    sam_string += "255";
    sam_string += SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped)
    //soft clip at begin of read
    if (parasail_result_is_sw(parasail_result_ptr)) {
        if (parasail_cigar_ptr->beg_query > 0) {
            sam_string += std::to_string(parasail_cigar_ptr->beg_query);
            sam_string += 'S';
        }
    }
    int editDistance = 0;

    for(int c=0; c < parasail_cigar_ptr->len; c++) {
        char letter = parasail_cigar_decode_op(parasail_cigar_ptr->seq[c]);
        uint32_t length = parasail_cigar_decode_len(parasail_cigar_ptr->seq[c]);
        sam_string += std::to_string(length);
        sam_string += letter;
        if ('X' == letter || 'I' == letter || 'D' == letter) {
            editDistance += length;
        }
    }
    if (parasail_result_is_sw(parasail_result_ptr)) {
        const int remainderInRead = readSequence.size() - parasail_result_get_end_query(parasail_result_ptr) - 1;
        //soft clip at end of read
        if (remainderInRead > 0) {
            sam_string += std::to_string(remainderInRead);
            sam_string += 'S';
        }
    }

    sam_string += SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    sam_string += '*';
    sam_string += SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    sam_string += readSequence;
    sam_string += SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        sam_string += '*';
        sam_string += SAM::SAM_SEPARATOR();
    }else{
        sam_string += readQuality;
        sam_string += SAM::SAM_SEPARATOR();
    }

    // OPTIONAL FIELDS

    //score
    sam_string += "AS:i:";
    sam_string += std::to_string(parasail_result_get_score(parasail_result_ptr));
    sam_string += SAM::SAM_SEPARATOR();

    //edit distance
    sam_string += "NM:i:";
    sam_string += std::to_string(editDistance);

    sam_string += '\n';
}

void writeAlignmentResultInSAMformat_NW_as_SW(
    std::ostream& stream,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view /*referenceSequence*/,
    std::string_view readQuality,
    int SWreadBeginPos,
    int SWreadEndPos_excl,
    int SWrefBeginPos,
    int /*SWrefEndPos_excl*/,
    const ParasailResultData& NWresultWithTraceback,
    const parasail_matrix_t* /*scoringmatrix*/
){
    parasail_result_t* parasail_result_ptr = NWresultWithTraceback.result.get();
    parasail_cigar_t* parasail_cigar_ptr = NWresultWithTraceback.cigar.get();

    // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
    //     parasail_result_get_cigar(
    //         parasail_result_ptr,
    //         readSequence.data() + SWreadBeginPos, 
    //         SWreadEndPos_excl - SWreadBeginPos,
    //         referenceSequence.data() + SWrefBeginPos, 
    //         SWrefEndPos_excl - SWrefBeginPos,
    //         scoringmatrix
    //     ),
    //     parasail_cigar_free
    // );


    // auto parasailCigarString = std::unique_ptr<char, void (*)(void*)>(
    //     parasail_cigar_decode(parasailCigar.get()),
    //     free
    // );

   
    stream << readName << SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = 0;
    // if(secondary){
    //     flags |= SAM::SAMFLAGS_SECONDARY();
    // }
    stream << flags << SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    stream << referenceName << SAM::SAM_SEPARATOR();

    //POS (1-based, 0 == unmapped)
    stream << SWrefBeginPos + 1 << SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    stream << 255 << SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped){

    //soft clip at begin of read
    if (SWreadBeginPos > 0) {
        stream << SWreadBeginPos << 'S';
    }

    int editDistance = 0;

    for(int c=0; c < parasail_cigar_ptr->len; c++) {
        char letter = parasail_cigar_decode_op(parasail_cigar_ptr->seq[c]);
        uint32_t length = parasail_cigar_decode_len(parasail_cigar_ptr->seq[c]);
        stream << length << letter;
        if ('X' == letter || 'I' == letter || 'D' == letter) {
            editDistance += length;
        }
    }

    const int remainderInRead = readSequence.size() - SWreadEndPos_excl;
    //soft clip at end of read
    if (remainderInRead > 0) {
        stream << remainderInRead << 'S';
    }

    stream << SAM::SAM_SEPARATOR();
    // stream << parasailCigarString.get() << SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    stream << '*' << SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    stream << 0 << SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    stream << 0 << SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    stream << readSequence << SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        stream << '*' << SAM::SAM_SEPARATOR();
    }else{
        stream << readQuality << SAM::SAM_SEPARATOR();
    }

    // OPTIONAL FIELDS

    //score
    stream << "AS:i:" << parasail_result_get_score(parasail_result_ptr) << SAM::SAM_SEPARATOR();

    //edit distance
    stream << "NM:i:" << editDistance;

    stream << '\n';

}

void appendAlignmentResultInSAMformat_NW_to_SW(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view /*referenceSequence*/,
    std::string_view readQuality,
    int SWreadBeginPos,
    int SWreadEndPos_excl,
    int SWrefBeginPos,
    int /*SWrefEndPos_excl*/,
    const ParasailResultData& NWresultWithTraceback,
    const parasail_matrix_t* /*scoringmatrix*/
){
    parasail_result_t* parasail_result_ptr = NWresultWithTraceback.result.get();
    parasail_cigar_t* parasail_cigar_ptr = NWresultWithTraceback.cigar.get();

    // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
    //     parasail_result_get_cigar(
    //         parasail_result_ptr,
    //         readSequence.data() + SWreadBeginPos, 
    //         SWreadEndPos_excl - SWreadBeginPos,
    //         referenceSequence.data() + SWrefBeginPos, 
    //         SWrefEndPos_excl - SWrefBeginPos,
    //         scoringmatrix
    //     ),
    //     parasail_cigar_free
    // );


    // auto parasailCigarString = std::unique_ptr<char, void (*)(void*)>(
    //     parasail_cigar_decode(parasailCigar.get()),
    //     free
    // );

    sam_string += readName;
    sam_string += SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = 0;
    sam_string += std::to_string(flags);
    sam_string += SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    sam_string += referenceName;
    sam_string += SAM::SAM_SEPARATOR();
    
    //POS (1-based, 0 == unmapped)
    sam_string += std::to_string(SWrefBeginPos + 1);
    sam_string += SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    sam_string += "255";
    sam_string += SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped){
    //soft clip at begin of read
    if (SWreadBeginPos > 0) {
        sam_string += std::to_string(SWreadBeginPos);
        sam_string += 'S';
    }

    int editDistance = 0;

    for(int c=0; c < parasail_cigar_ptr->len; c++) {
        char letter = parasail_cigar_decode_op(parasail_cigar_ptr->seq[c]);
        uint32_t length = parasail_cigar_decode_len(parasail_cigar_ptr->seq[c]);
        sam_string += std::to_string(length);
        sam_string += letter;
        if ('X' == letter || 'I' == letter || 'D' == letter) {
            editDistance += length;
        }
    }

    const int remainderInRead = readSequence.size() - SWreadEndPos_excl;
    //soft clip at end of read
    if (remainderInRead > 0) {
        sam_string += std::to_string(remainderInRead);
        sam_string += 'S';
    }

    sam_string += SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    sam_string += '*';
    sam_string += SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    sam_string += readSequence;
    sam_string += SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        sam_string += '*';
        sam_string += SAM::SAM_SEPARATOR();
    }else{
        sam_string += readQuality;
        sam_string += SAM::SAM_SEPARATOR();
    }

    // OPTIONAL FIELDS

    //score
    sam_string += "AS:i:";
    sam_string += std::to_string(parasail_result_get_score(parasail_result_ptr));
    sam_string += SAM::SAM_SEPARATOR();

    //edit distance
    sam_string += "NM:i:";
    sam_string += std::to_string(editDistance);

    sam_string += '\n';
}


void appendAlignmentResultInSAMformat_SG(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view /*referenceSequence*/,
    std::string_view readQuality,
    const ParasailResultData& alignmentResultWithTraceback,
    const parasail_matrix_t* /*scoringmatrix*/
){

    parasail_result_t* parasail_result_ptr = alignmentResultWithTraceback.result.get();
    parasail_cigar_t* parasail_cigar_ptr = alignmentResultWithTraceback.cigar.get();

    // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
    //     parasail_result_get_cigar(
    //         alignmentResultWithTraceback,
    //         readSequence.data(), readSequence.size(),
    //         referenceSequence.data(), referenceSequence.size(),
    //         scoringmatrix
    //     ),
    //     parasail_cigar_free
    // );

    sam_string += readName;
    sam_string += SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = 0;
    sam_string += std::to_string(flags);
    sam_string += SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    sam_string += referenceName;
    sam_string += SAM::SAM_SEPARATOR();

    //POS (1-based, 0 == unmapped)
    sam_string += std::to_string(parasail_cigar_ptr->beg_ref + 1);
    sam_string += SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    sam_string += "255";
    sam_string += SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped)
    //soft clip at begin of read
    // if (parasail_result_is_sw(parasail_result_ptr)) {
    //     if (parasail_cigar_ptr->beg_query > 0) {
    //         sam_string += std::to_string(parasail_cigar_ptr->beg_query);
    //         sam_string += 'S';
    //     }
    // }
    int editDistance = 0;

    for(int c=0; c < parasail_cigar_ptr->len; c++) {
        char letter = parasail_cigar_decode_op(parasail_cigar_ptr->seq[c]);
        uint32_t length = parasail_cigar_decode_len(parasail_cigar_ptr->seq[c]);
        sam_string += std::to_string(length);
        sam_string += letter;
        if(c == 0 || c == parasail_cigar_ptr->len-1){
            //do not count gaps at begin and end
            if ('X' == letter) {
                editDistance += length;
            }
        }else{
            if ('X' == letter || 'I' == letter || 'D' == letter) {
                editDistance += length;
            }
        }
    }
    // if (parasail_result_is_sw(parasail_result_ptr)) {
    //     const int remainderInRead = readSequence.size() - parasail_result_get_end_query(parasail_result_ptr) - 1;
    //     //soft clip at end of read
    //     if (remainderInRead > 0) {
    //         sam_string += std::to_string(remainderInRead);
    //         sam_string += 'S';
    //     }
    // }

    sam_string += SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    sam_string += '*';
    sam_string += SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    sam_string += readSequence;
    sam_string += SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        sam_string += '*';
        sam_string += SAM::SAM_SEPARATOR();
    }else{
        sam_string += readQuality;
        sam_string += SAM::SAM_SEPARATOR();
    }

    // OPTIONAL FIELDS

    //score
    sam_string += "AS:i:";
    sam_string += std::to_string(parasail_result_get_score(parasail_result_ptr));
    sam_string += SAM::SAM_SEPARATOR();

    //edit distance
    sam_string += "NM:i:";
    sam_string += std::to_string(editDistance);

    sam_string += '\n';
}

void appendAlignmentResultInSAMformat_NW_to_SG(
    std::string& sam_string,
    std::string_view readName, 
    std::string_view referenceName,
    std::string_view readSequence,
    std::string_view /*referenceSequence*/,
    std::string_view readQuality,
    int SGreadBeginPos,
    int SGreadEndPos_excl,
    int SGrefBeginPos,
    int /*SGrefEndPos_excl*/,
    int score,
    const ParasailResultData& NWresultWithTraceback,
    const parasail_matrix_t* /*scoringmatrix*/
){
    //parasail_result_t* parasail_result_ptr = NWresultWithTraceback.result.get();
    parasail_cigar_t* parasail_cigar_ptr = NWresultWithTraceback.cigar.get();

    // auto parasailCigar = std::unique_ptr<parasail_cigar_t, void (*) (parasail_cigar_t*)>(
    //     parasail_result_get_cigar(
    //         parasail_result_ptr,
    //         readSequence.data() + SWreadBeginPos, 
    //         SWreadEndPos_excl - SWreadBeginPos,
    //         referenceSequence.data() + SWrefBeginPos, 
    //         SWrefEndPos_excl - SWrefBeginPos,
    //         scoringmatrix
    //     ),
    //     parasail_cigar_free
    // );


    // auto parasailCigarString = std::unique_ptr<char, void (*)(void*)>(
    //     parasail_cigar_decode(parasailCigar.get()),
    //     free
    // );


    sam_string += readName;
    sam_string += SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = 0;
    sam_string += std::to_string(flags);
    sam_string += SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    sam_string += referenceName;
    sam_string += SAM::SAM_SEPARATOR();
    
    
    //POS (1-based, 0 == unmapped)
    const char cigar_letter0 = parasail_cigar_decode_op(parasail_cigar_ptr->seq[0]);
    const uint32_t cigar_length0 = parasail_cigar_decode_len(parasail_cigar_ptr->seq[0]);
    const int POS = 1 + SGrefBeginPos + (cigar_letter0 == 'D' ? cigar_length0 : 0);
    sam_string += std::to_string(POS);
    sam_string += SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    sam_string += "255";
    sam_string += SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped){

    int editDistance = 0;

    if (SGreadBeginPos > 0) {
        sam_string += std::to_string(SGreadBeginPos);
        sam_string += 'I';
    }

    for(int c=0; c < parasail_cigar_ptr->len; c++) {
        char letter = parasail_cigar_decode_op(parasail_cigar_ptr->seq[c]);
        uint32_t length = parasail_cigar_decode_len(parasail_cigar_ptr->seq[c]);
        //first, but not last
        if(c == 0 && c != parasail_cigar_ptr->len-1){
            //do not count gaps at begin
            if ('X' == letter) {
                editDistance += length;
            }
            // do not output D at begin. we already shifted the ref start POS accordingly
            if ('D' != letter){
                sam_string += std::to_string(length);
                sam_string += letter;
            }
        }else if(c == 0 && c == parasail_cigar_ptr->len-1){
            //first and last

            // do not output D at begin. we already shifted the ref start POS accordingly
            if ('D' != letter){
                sam_string += std::to_string(length);
                sam_string += letter;
            }

            //do not count gaps at end
            if ('X' == letter) {
                editDistance += length;
            }
            //manually add insertions at the end if read reaches beyond reference and insertion is missing from the cigar
            if ('I' != letter){
                const int remainder = readSequence.size() - SGreadEndPos_excl;
                if(remainder > 0){
                    sam_string += std::to_string(remainder);
                    sam_string += 'I';
                }
            }
        }else if(c != 0 && c == parasail_cigar_ptr->len-1){
            //not first, but last
            sam_string += std::to_string(length);
            sam_string += letter;

            //do not count gaps at end
            if ('X' == letter) {
                editDistance += length;
            }
            //manually add insertions at the end if read reaches beyond reference and insertion is missing from the cigar
            if ('I' != letter){
                const int remainder = readSequence.size() - SGreadEndPos_excl;
                if(remainder > 0){
                    sam_string += std::to_string(remainder);
                    sam_string += 'I';
                }
            }
        }else{
            if ('X' == letter || 'I' == letter || 'D' == letter) {
                editDistance += length;
            }
            sam_string += std::to_string(length);
            sam_string += letter;
        }
    }

    // const int remainderInRead = readSequence.size() - SWreadEndPos_excl;
    // //soft clip at end of read
    // if (remainderInRead > 0) {
    //     sam_string += std::to_string(remainderInRead);
    //     sam_string += 'S';
    // }

    sam_string += SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    sam_string += '*';
    sam_string += SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    sam_string += '0';
    sam_string += SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    sam_string += readSequence;
    sam_string += SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        sam_string += '*';
        sam_string += SAM::SAM_SEPARATOR();
    }else{
        sam_string += readQuality;
        sam_string += SAM::SAM_SEPARATOR();
    }

    // OPTIONAL FIELDS

    //score
    sam_string += "AS:i:";
    sam_string += std::to_string(score);
    sam_string += SAM::SAM_SEPARATOR();

    //edit distance
    sam_string += "NM:i:";
    sam_string += std::to_string(editDistance);

    sam_string += '\n';
}



void writeUnmappedAlignmentResultInSAMformat(
    std::ostream& stream,
    std::string_view readName, 
    std::string_view readSequence,
    std::string_view readQuality
){
    stream << readName << SAM::SAM_SEPARATOR();

    //FLAG
    unsigned int flags = SAM::SAMFLAGS_UNMAP();
    // if(secondary){
    //     flags |= SAM::SAMFLAGS_SECONDARY();
    // }
    stream << flags << SAM::SAM_SEPARATOR();

    //RNAME (chromosome name, * == unmapped)
    stream << "*" << SAM::SAM_SEPARATOR();

    //POS (1-based, 0 == unmapped)
    stream << 0 << SAM::SAM_SEPARATOR();

    //MAPQ (255 == not available)
    stream << 255 << SAM::SAM_SEPARATOR();

    //CIGAR (* == unmapped)
    stream << "*" << SAM::SAM_SEPARATOR();

    //RNEXT (* == not available)
    stream << '*' << SAM::SAM_SEPARATOR();

    //PNEXT (0 == not available)
    stream << 0 << SAM::SAM_SEPARATOR();

    //TLEN (0 == not available)
    stream << 0 << SAM::SAM_SEPARATOR();

    //SEQ (* == not available)
    stream << readSequence << SAM::SAM_SEPARATOR();

    //QUAL (* == not available)
    if(readQuality.size() == 0){
        stream << '*' << SAM::SAM_SEPARATOR();
    }else{
        stream << readQuality << SAM::SAM_SEPARATOR();
    }


    stream << '\n';
}