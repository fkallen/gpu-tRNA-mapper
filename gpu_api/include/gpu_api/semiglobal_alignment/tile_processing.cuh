#ifndef SEMIGLOBAL_ALIGNMENT_TILE_PROCESSING_CUH
#define SEMIGLOBAL_ALIGNMENT_TILE_PROCESSING_CUH

#include "state_common.cuh"
#include "../util.cuh"

#include <cstdint>

#define MYFORCEINLINE __forceinline__
// #define MYFORCEINLINE

namespace semiglobalalignment{

    template<class Group, class TempType>
    struct TempHandler{
    public:
        __device__
        TempHandler(Group& g,  TempType* ptr) 
            : tempWriteOffset(g.thread_rank()), 
            tempLoadOffset(g.thread_rank()), 
            group(g), 
            groupTempStorage(ptr){}

        template<class LastColumn>
        __device__
        void save(const LastColumn& lastColumn){
            #if 0
            // printf("B tid %d, write %f\n", group.thread_rank(), lastColumn.M);
            printf("tid %d, write (%d %d) %d\n", group.thread_rank(), 
                int(lastColumn.getPayload().x), int(lastColumn.getPayload().y), tempWriteOffset);
            // printf("tid %d, write (%f %f) (%f %f)\n", group.thread_rank(), 
            //     float(lastColumn.getPayload().x.x), float(lastColumn.getPayload().x.y),
            //     float(lastColumn.getPayload().y.x), float(lastColumn.getPayload().y.y)
            // );
            #endif
            groupTempStorage[tempWriteOffset] = lastColumn.getPayload();
            tempWriteOffset += group.size();
        }

        template<class LastColumn>
        __device__
        void finalSave(const LastColumn& lastColumn, int firstValidThread){
            #if 0
            // printf("B tid %d, write %f\n", group.thread_rank(), lastColumn.M);
            printf("tid %d, write final (%d %d) %d\n", group.thread_rank(), 
                int(lastColumn.getPayload().x), int(lastColumn.getPayload().y), tempWriteOffset - firstValidThread);
            // printf("tid %d, write (%f %f) (%f %f)\n", group.thread_rank(), 
            //     float(lastColumn.getPayload().x.x), float(lastColumn.getPayload().x.y),
            //     float(lastColumn.getPayload().y.x), float(lastColumn.getPayload().y.y)
            // );
            #endif
            groupTempStorage[tempWriteOffset - firstValidThread] = lastColumn.getPayload();
        }

        __device__
        TempType load(){
            TempType val = groupTempStorage[tempLoadOffset];
            #if 0
            // printf("tid %d, load %f\n", group.thread_rank(), groupTempStorage[tempLoadOffset]);
            printf("tid %d, load (%f %f) %d\n", group.thread_rank(), float(val.x), float(val.y), tempLoadOffset);
            // printf("tid %d, load (%f %f) (%f %f)\n", group.thread_rank(), 
            //     float(val.x.x), float(val.x.y),
            //     float(val.y.x), float(val.y.y)
            // );
            #endif
            tempLoadOffset += group.size();
            return val;
        }

    private:
        int tempWriteOffset;
        int tempLoadOffset;
        Group& group;
        TempType* const groupTempStorage;
    };

    template<class Group, class State, class SubjectLettersData, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processSingleTile_fromStart(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback
    ){
        int r = 1;
        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileFirstDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        // process rows which do not cover the last valid row. no lastRowCallback required        
        for(; r < numRows - int(group.size()) - 3; r += 4){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
        }

        //process remaining rows
        for(; r < numRows - 3; r += 4){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processSingleTile_continued(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r,
        int numRows,
        LastRowCallback& lastRowCallback
    ){
        if(r == 1 && r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileFirstDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        {
            //process up to three letters such that we leave this section after processing letter <3>
            int leftoverIn4 = 0;
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }

            //leftover 1: process <3>
            //leftover 2: process <2> <3>
            //leftover 3: process <1> <2> <3>
            //leftover 0: do nothing
            
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 1 && r < numRows){
                //process <3>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
                lastRowCallback(r);
                r++;
            }
        }

        // process rows which do not cover the last valid row. no lastRowCallback required        
        for(; r < numRows - int(group.size()) - 3; r += 4){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
        }

        //process remaining rows
        for(; r < numRows - 3; r += 4){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processSingleTile_alwayscheck_fromStart(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback
    ){
        int r = 1;
        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileFirstDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        //process remaining rows
        for(; r < numRows - 3; r += 4){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processSingleTile_alwayscheck_continued(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r,
        int numRows,
        LastRowCallback& lastRowCallback
    ){
        if(r == 1 && r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileFirstDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        {
            //process up to three letters such that we leave this section after processing letter <3>
            int leftoverIn4 = 0;
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }

            //leftover 1: process <3>
            //leftover 2: process <2> <3>
            //leftover 3: process <1> <2> <3>
            //leftover 0: do nothing
            
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 1 && r < numRows){
                //process <3>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
                lastRowCallback(r);
                r++;
            }
        }

        //process remaining rows
        for(; r < numRows - 3; r += 4){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processFirstTile_fromStart(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);


        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
        int r = 1;
        NoLastColumn noLastColumn;
        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
            state.stepFirstTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        for(; r < min(group.size(), numRows);){
            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter(); 
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters(); 
    
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }
        }

        // process rows which do not cover the last valid row. no lastRowCallback required
        for(; r < numRows - int(group.size()) - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            //for vector types we still need to check for last row. this way we can just process it in one go using the larger number of rows as limit
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processFirstTile_continued(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r, // as returned from processFirstTile_fromStart or processFirstTile_continued
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);


        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
        if(r < group.size()){
            NoLastColumn noLastColumn;
            if(r == 1 && r < numRows){
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
                state.stepFirstTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }
            if(r == 2 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }
            if(r == 3 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            const int rowLimit = min(group.size(), numRows);
            {
                //process up to three letters such that we leave this section after processing letter <2>
                int leftoverIn4 = 0;
                // if(r%4 == 1){ leftoverInChar4 = 0; }
                // else 
                if(r%4 == 2){ leftoverIn4 = 3; }
                else if(r%4 == 3){ leftoverIn4 = 2; }
                else if(r%4 == 0){ leftoverIn4 = 1; }
                // printf("line %d, r %d, leftoverIn4 %d\n", __LINE__, r, leftoverIn4);

                //leftover 1: do nothing
                //leftover 2: process <2>
                //leftover 3: process <1>, <2>
                //leftover 0: reload, then process <0>, <1>, <2>
                
                if(leftoverIn4 == 0 && r < rowLimit){
                    //process <0>
                    subjectLetters.shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        subjectLetters.loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        subjectLetters.shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                    leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
                }
                if(leftoverIn4 >= 3 && r < rowLimit){
                    //process <1>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
                if(leftoverIn4 >= 2 && r < rowLimit){
                    //process <2>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }

            for(; r < rowLimit;){
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
    
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter(); 
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters(); 
        
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
    
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
    
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }
        }

        if(r < numRows){
            //process up to three letters such that we leave this section after processing letter <2>
            int leftoverIn4 = 0;
            // if(r%4 == 1){ leftoverInChar4 = 0; }
            // else 
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }
            // printf("line %d, r %d, leftoverIn4 %d\n", __LINE__, r, leftoverIn4);

            //leftover 1: do nothing
            //leftover 2: process <2>
            //leftover 3: process <1>, <2>
            //leftover 0: reload, then process <0>, <1>, <2>
            
            if(leftoverIn4 == 0 && r < numRows){
                //process <0>
                subjectLetters.shuffleCurrentLetter();
                if((r-1) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    subjectLetters.loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters();
                }
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
                lastRowCallback(r);
                r++;
                leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
            }
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
                lastRowCallback(r);
                // if(group.thread_rank() == 0){
                //     printf("r %d, tempHandler.save ? %d\n", r, (r+1) % (group.size()) == 0);
                // }
                if((r+1) % (group.size()) == 0){
                    tempHandler.save(lastColumn);
                }  
                r++;
            }
        }

        // process rows which do not cover the last valid row. no lastRowCallback required
        for(; r < numRows - int(group.size()) - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            //for vector types we still need to check for last row. this way we can just process it in one go using the larger number of rows as limit
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processFirstTile_alwayscheck_fromStart(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);


        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
        int r = 1;
        NoLastColumn noLastColumn;
        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
            state.stepFirstTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        for(; r < min(group.size(), numRows);){
            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter(); 
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters(); 
    
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }
        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processFirstTile_alwayscheck_continued(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r, // as returned from processFirstTile_fromStart or processFirstTile_continued
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);


        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
        if(r < group.size()){
            NoLastColumn noLastColumn;
            if(r == 1 && r < numRows){
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
                state.stepFirstTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }
            if(r == 2 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }
            if(r == 3 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            const int rowLimit = min(group.size(), numRows);
            {
                //process up to three letters such that we leave this section after processing letter <2>
                int leftoverIn4 = 0;
                // if(r%4 == 1){ leftoverInChar4 = 0; }
                // else 
                if(r%4 == 2){ leftoverIn4 = 3; }
                else if(r%4 == 3){ leftoverIn4 = 2; }
                else if(r%4 == 0){ leftoverIn4 = 1; }
                // printf("line %d, r %d, leftoverIn4 %d\n", __LINE__, r, leftoverIn4);

                //leftover 1: do nothing
                //leftover 2: process <2>
                //leftover 3: process <1>, <2>
                //leftover 0: reload, then process <0>, <1>, <2>
                
                if(leftoverIn4 == 0 && r < rowLimit){
                    //process <0>
                    subjectLetters.shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        subjectLetters.loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        subjectLetters.shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                    leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
                }
                if(leftoverIn4 >= 3 && r < rowLimit){
                    //process <1>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
                if(leftoverIn4 >= 2 && r < rowLimit){
                    //process <2>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }

            for(; r < rowLimit;){
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
    
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter(); 
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters(); 
        
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
    
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
    
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }
        }

        if(r < numRows){
            //process up to three letters such that we leave this section after processing letter <2>
            int leftoverIn4 = 0;
            // if(r%4 == 1){ leftoverInChar4 = 0; }
            // else 
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }
            // printf("line %d, r %d, leftoverIn4 %d\n", __LINE__, r, leftoverIn4);

            //leftover 1: do nothing
            //leftover 2: process <2>
            //leftover 3: process <1>, <2>
            //leftover 0: reload, then process <0>, <1>, <2>
            
            if(leftoverIn4 == 0 && r < numRows){
                //process <0>
                subjectLetters.shuffleCurrentLetter();
                if((r-1) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    subjectLetters.loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters();
                }
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
                lastRowCallback(r);
                r++;
                leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
            }
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
                lastRowCallback(r);
                // if(group.thread_rank() == 0){
                //     printf("r %d, tempHandler.save ? %d\n", r, (r+1) % (group.size()) == 0);
                // }
                if((r+1) % (group.size()) == 0){
                    tempHandler.save(lastColumn);
                }  
                r++;
            }
        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }


    template<class Group, class State, class SubjectLettersData, class LeftBorder, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processIntermediateTile_fromStart(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);

        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row

        int r = 1;
        NoLastColumn noLastColumn;
        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        for(; r < min(group.size(), numRows);){
            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter(); 
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
                
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }
        }

       

        // process rows which do not cover the last valid row. no lastRowCallback required
        for(; r < numRows - int(group.size()) - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            //for vector types we still need to check for last row. this way we can just process it in one go using the larger number of rows as limit
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if(r % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r+1, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if(r % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r+1, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }


    template<class Group, class State, class SubjectLettersData, class LeftBorder, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processIntermediateTile_continued(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r, // as returned from processIntermediateTile_fromStart or processIntermediateTile_continued
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);

        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row

        if(r < group.size()){
            NoLastColumn noLastColumn;
            if(r == 1 && r < numRows){
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r == 2 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r == 3 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            const int rowLimit = min(group.size(), numRows);
            {
                //process up to three letters such that we leave this section after processing letter <2>
                int leftoverIn4 = 0;
                // if(r%4 == 1){ leftoverInChar4 = 0; }
                // else 
                if(r%4 == 2){ leftoverIn4 = 3; }
                else if(r%4 == 3){ leftoverIn4 = 2; }
                else if(r%4 == 0){ leftoverIn4 = 1; }

                //leftover 1: do nothing
                //leftover 2: process <2>
                //leftover 3: process <1>, <2>
                //leftover 0: reload, then process <0>, <1>, <2>
                
                if(leftoverIn4 == 0 && r < numRows){
                    //process <0>
                    subjectLetters.shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        subjectLetters.loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        subjectLetters.shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    if((r-1) % group.size() == 0 && r <= subjectLength){
                        leftBorder.setPayload(tempHandler.load());
                        if(group.thread_rank() == 0){
                            state.updateFromLeftBorder(r, leftBorder);
                        }
                    }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                    lastRowCallback(r);
                    r++;
                    leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
                }
                if(leftoverIn4 >= 3 && r < numRows){
                    //process <1>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                    lastRowCallback(r);
                    r++;
                }
                if(leftoverIn4 >= 2 && r < numRows){
                    //process <2>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }
            for(; r < rowLimit;){
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }

                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter(); 
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters();
                    
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }

                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }

                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }
        }

        if(r != group.size()){
            //process up to three letters such that we leave this section after processing letter <2>
            int leftoverIn4 = 0;
            // if(r%4 == 1){ leftoverInChar4 = 0; }
            // else 
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }

            //leftover 1: do nothing
            //leftover 2: process <2>
            //leftover 3: process <1>, <2>
            //leftover 0: reload, then process <0>, <1>, <2>
            
            if(leftoverIn4 == 0 && r < numRows){
                //process <0>
                subjectLetters.shuffleCurrentLetter();
                if((r-1) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    subjectLetters.loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters();
                }
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                if((r-1) % group.size() == 0 && r <= subjectLength){
                    leftBorder.setPayload(tempHandler.load());
                    if(group.thread_rank() == 0){
                        state.updateFromLeftBorder(r, leftBorder);
                    }
                }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                lastRowCallback(r);
                r++;
                leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
            }
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                lastRowCallback(r);
                if((r+1) % (group.size()) == 0){
                    tempHandler.save(lastColumn);
                }  
                r++;
            }
        }

        // process rows which do not cover the last valid row. no lastRowCallback required
        for(; r < numRows - int(group.size()) - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            //for vector types we still need to check for last row. this way we can just process it in one go using the larger number of rows as limit
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if(r % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r+1, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if(r % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r+1, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LeftBorder, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processIntermediateTile_alwayscheck_fromStart(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);

        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row

        int r = 1;
        NoLastColumn noLastColumn;
        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
            lastRowCallback(r);
            r++;
        }

        for(; r < min(group.size(), numRows);){
            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter(); 
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
                
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }
        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if(r % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r+1, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }


    template<class Group, class State, class SubjectLettersData, class LeftBorder, class LastColumn, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processIntermediateTile_alwayscheck_continued(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r, // as returned from processIntermediateTile_fromStart or processIntermediateTile_continued
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);

        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row

        if(r < group.size()){
            NoLastColumn noLastColumn;
            if(r == 1 && r < numRows){
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r == 2 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            if(r == 3 && r < numRows){
                subjectLetters.shuffleCurrentLetter();
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                lastRowCallback(r);
                r++;
            }

            const int rowLimit = min(group.size(), numRows);
            {
                //process up to three letters such that we leave this section after processing letter <2>
                int leftoverIn4 = 0;
                // if(r%4 == 1){ leftoverInChar4 = 0; }
                // else 
                if(r%4 == 2){ leftoverIn4 = 3; }
                else if(r%4 == 3){ leftoverIn4 = 2; }
                else if(r%4 == 0){ leftoverIn4 = 1; }

                //leftover 1: do nothing
                //leftover 2: process <2>
                //leftover 3: process <1>, <2>
                //leftover 0: reload, then process <0>, <1>, <2>
                
                if(leftoverIn4 == 0 && r < numRows){
                    //process <0>
                    subjectLetters.shuffleCurrentLetter();
                    if((r-1) % (4*group.size()) == 0){
                        //used up all query letters stored across the group. reload
                        subjectLetters.loadNext4Letters();
                    }else{
                        //get next 4 letters from neighbor
                        subjectLetters.shuffle4Letters();
                    }
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    if((r-1) % group.size() == 0 && r <= subjectLength){
                        leftBorder.setPayload(tempHandler.load());
                        if(group.thread_rank() == 0){
                            state.updateFromLeftBorder(r, leftBorder);
                        }
                    }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                    lastRowCallback(r);
                    r++;
                    leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
                }
                if(leftoverIn4 >= 3 && r < numRows){
                    //process <1>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                    lastRowCallback(r);
                    r++;
                }
                if(leftoverIn4 >= 2 && r < numRows){
                    //process <2>
                    subjectLetters.shuffleCurrentLetter(); 
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }
            for(; r < rowLimit;){
                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }

                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter(); 
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters();
                    
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }

                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }

                if(r < rowLimit){
                    subjectLetters.shuffleCurrentLetter();
                    if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                    leftBorder.shuffleDown(group);
                    state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);
                    lastRowCallback(r);
                    r++;
                }
            }
        }

        if(r != group.size()){
            //process up to three letters such that we leave this section after processing letter <2>
            int leftoverIn4 = 0;
            // if(r%4 == 1){ leftoverInChar4 = 0; }
            // else 
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }

            //leftover 1: do nothing
            //leftover 2: process <2>
            //leftover 3: process <1>, <2>
            //leftover 0: reload, then process <0>, <1>, <2>
            
            if(leftoverIn4 == 0 && r < numRows){
                //process <0>
                subjectLetters.shuffleCurrentLetter();
                if((r-1) % (4*group.size()) == 0){
                    //used up all query letters stored across the group. reload
                    subjectLetters.loadNext4Letters();
                }else{
                    //get next 4 letters from neighbor
                    subjectLetters.shuffle4Letters();
                }
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
                if((r-1) % group.size() == 0 && r <= subjectLength){
                    leftBorder.setPayload(tempHandler.load());
                    if(group.thread_rank() == 0){
                        state.updateFromLeftBorder(r, leftBorder);
                    }
                }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                lastRowCallback(r);
                r++;
                leftoverIn4 = 3; //continue processing the remainder of these new 4 letters
            }
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
                lastRowCallback(r);
                if((r+1) % (group.size()) == 0){
                    tempHandler.save(lastColumn);
                }  
                r++;
            }
        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if((r) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }                           
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if(r % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r+1, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+1);
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            lastRowCallback(r+3);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LeftBorder, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processLastTile_fromStart(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        TempHandler& tempHandler
    ){

        int r = 1;

        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        // process rows which do not cover the last valid row. no lastRowCallback required
        for(; r < numRows - int(group.size()) - 3; r += 4){        
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
        }

        for(; r < numRows - 3; r += 4){        
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); } 
            leftBorder.shuffleDown(group);  
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LeftBorder, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processLastTile_continued(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r, // as returned from processLastTile_fromStart or processLastTile_continued
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        TempHandler& tempHandler
    ){

        if(r == 1 && r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        {
            //process up to three letters such that we leave this section after processing letter <3>
            int leftoverIn4 = 0;
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }

            //leftover 1: process <3>
            //leftover 2: process <2> <3>
            //leftover 3: process <1> <2> <3>
            //leftover 0: do nothing
            
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 1 && r < numRows){
                //process <3>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                leftBorder.shuffleDown(group);
                state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
                lastRowCallback(r);
                r++;
            }
        }

        // process rows which do not cover the last valid row. no lastRowCallback required
        for(; r < numRows - int(group.size()) - 3; r += 4){        
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+1);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+2);
            }

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);
            if constexpr(std::is_same_v<half2, typename State::ScoreType> || std::is_same_v<short2, typename State::ScoreType>){
                lastRowCallback(r+3);
            }
        }

        for(; r < numRows - 3; r += 4){        
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); } 
            leftBorder.shuffleDown(group);  
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        return r;
    }


    template<class Group, class State, class SubjectLettersData, class LeftBorder, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processLastTile_alwayscheck_fromStart(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        TempHandler& tempHandler
    ){

        int r = 1;

        if(r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        for(; r < numRows - 3; r += 4){        
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); } 
            leftBorder.shuffleDown(group);  
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LeftBorder, class TempHandler, class LastRowCallback>
    __device__ MYFORCEINLINE
    int processLastTile_alwayscheck_continued(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r, // as returned from processLastTile_fromStart or processLastTile_continued
        int numRows,
        LastRowCallback& lastRowCallback,
        int subjectLength,
        LeftBorder& leftBorder,
        TempHandler& tempHandler
    ){

        if(r == 1 && r < numRows){
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        {
            //process up to three letters such that we leave this section after processing letter <3>
            int leftoverIn4 = 0;
            if(r%4 == 2){ leftoverIn4 = 3; }
            else if(r%4 == 3){ leftoverIn4 = 2; }
            else if(r%4 == 0){ leftoverIn4 = 1; }

            //leftover 1: process <3>
            //leftover 2: process <2> <3>
            //leftover 3: process <1> <2> <3>
            //leftover 0: do nothing
            
            if(leftoverIn4 >= 3 && r < numRows){
                //process <1>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
                leftBorder.shuffleDown(group);
                state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 2 && r < numRows){
                //process <2>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
                leftBorder.shuffleDown(group);
                state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
                lastRowCallback(r);
                r++;
            }
            if(leftoverIn4 >= 1 && r < numRows){
                //process <3>
                subjectLetters.shuffleCurrentLetter(); 
                if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
                leftBorder.shuffleDown(group);
                state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
                lastRowCallback(r);
                r++;
            }
        }

        for(; r < numRows - 3; r += 4){        
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);
            lastRowCallback(r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);
            lastRowCallback(r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);
            lastRowCallback(r+3);
        }

        //can have at most 3 remaining rows
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter(); 
            if((r-1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            }
            if((r-1) % group.size() == 0 && r <= subjectLength){
                leftBorder.setPayload(tempHandler.load());
                if(group.thread_rank() == 0){
                    state.updateFromLeftBorder(r, leftBorder);
                }
            }
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); } 
            leftBorder.shuffleDown(group);  
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            lastRowCallback(r);
            r++;
        }

        return r;
    }


    template<class Group, class LastColumn, class TempHandler>
    __device__ MYFORCEINLINE
    void tempStorageTileCompleted(
        Group& group,
        int numComputedNonOOBRowsInTile,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        const int totalChunksOfFour = (numComputedNonOOBRowsInTile) / 4;
        const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
        const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + numComputedNonOOBRowsInTile % 4;
        // if(threadIdx.x == 0){
        //     printf("totalChunksOfFour %d, unsavedChunksOfFour %d, numThreadsWithValidTileLastColumn %d, numComputedNonOOBRowsInTile %d\n", 
        //             totalChunksOfFour, unsavedChunksOfFour, numThreadsWithValidTileLastColumn, numComputedNonOOBRowsInTile);
        // }
        if(numThreadsWithValidTileLastColumn > 0){
            const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
            if(group.thread_rank() >= firstValidThread){
                tempHandler.finalSave(lastColumn, firstValidThread);
            }
        }
    }

    template<class Group, class LastColumn, class TempHandler>
    __device__ MYFORCEINLINE
    void tempStorageTileCompleted_half2OrShort2_oneAlignmentFinished(
        Group& group,
        int numComputedNonOOBRowsInTile_maybefinished,
        int numComputedRowsInTile_maybefinished,
        int numComputedRowsInTile_finished,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        const int totalChunksOfFour_maybefinished = (numComputedNonOOBRowsInTile_maybefinished) / 4;
        const int unsavedChunksOfFour_maybefinished = totalChunksOfFour_maybefinished % (group.size() / 4);
        const int numThreadsWithValidTileLastColumn_maybefinished = unsavedChunksOfFour_maybefinished * 4 + numComputedNonOOBRowsInTile_maybefinished % 4;

        // if(threadIdx.x == 0){
        //     printf("totalChunksOfFour_maybefinished %d, unsavedChunksOfFour_maybefinished %d, "
        //         "numThreadsWithValidTileLastColumn_maybefinished %d, numComputedNonOOBRowsInTile_maybefinished %d, "
        //         "numComputedRowsInTile_maybefinished %d, numComputedRowsInTile_finished %d\n", 
        //             totalChunksOfFour_maybefinished, unsavedChunksOfFour_maybefinished, 
        //             numThreadsWithValidTileLastColumn_maybefinished, numComputedNonOOBRowsInTile_maybefinished,
        //             numComputedRowsInTile_maybefinished, numComputedRowsInTile_finished
        //     );
        // }
        if(numThreadsWithValidTileLastColumn_maybefinished > 0){
            if(numComputedRowsInTile_maybefinished >= numComputedRowsInTile_finished){
                const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn_maybefinished;
                if(group.thread_rank() >= firstValidThread){
                    tempHandler.finalSave(lastColumn, firstValidThread);
                }
            }else{
                const int difference = numComputedRowsInTile_finished - numComputedRowsInTile_maybefinished;
                const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn_maybefinished - difference;
                // printf("line %d, tid %lu, difference %d, firstValidThread %d\n", __LINE__, cooperative_groups::this_grid().thread_rank(), difference, firstValidThread);
                if(group.thread_rank() >= firstValidThread && group.thread_rank() < firstValidThread + numThreadsWithValidTileLastColumn_maybefinished){
                    // printf("tid %d will save with difference %d, firstValidThread %d\n", group.thread_rank(), difference, firstValidThread);
                    tempHandler.finalSave(lastColumn, firstValidThread);
                }
            }
        }
    }


}

#ifdef MYFORCEINLINE
#undef MYFORCEINLINE
#endif

#endif