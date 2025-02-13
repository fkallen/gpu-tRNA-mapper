#ifndef LOCAL_ALIGNMENT_TILE_PROCESSING_CUH
#define LOCAL_ALIGNMENT_TILE_PROCESSING_CUH

#include "state_common.cuh"
#include "../util.cuh"

#include <cstdint>

namespace localalignment{


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
            printf("tid %d, write (%f %f)\n", group.thread_rank(), float(lastColumn.M.x), float(lastColumn.M.y));
            #endif
            groupTempStorage[tempWriteOffset] = lastColumn.getPayload();
            tempWriteOffset += group.size();
        }

        template<class LastColumn>
        __device__
        void finalSave(const LastColumn& lastColumn, int firstValidThread){
            #if 0
            // printf("B tid %d, write %f\n", group.thread_rank(), lastColumn.M);
            printf("tid %d, write (%f %f)\n", group.thread_rank(), float(lastColumn.getPayload().x), float(lastColumn.getPayload().y));
            #endif
            groupTempStorage[tempWriteOffset - firstValidThread] = lastColumn.getPayload();
        }

        __device__
        TempType load(){
            TempType val = groupTempStorage[tempLoadOffset];
            #if 0
            // printf("tid %d, load %f\n", group.thread_rank(), groupTempStorage[tempLoadOffset]);
            printf("tid %d, load (%f %f)\n", group.thread_rank(), float(val.x), float(val.y));
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

    template<class Group, class State, class SubjectLettersData>
    __device__ __forceinline__
    int processSingleTile(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r,
        int numRows
    ){
        if(r > 1){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
                //used up all query letters stored across the group. reload
                subjectLetters.loadNext4Letters();
            }else{
                //get next 4 letters from neighbor
                subjectLetters.shuffle4Letters();
            } 
        }
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
        if(r == 1){
            state.stepSingleTileFirstDiagonal(subjectLetters.getCurrentLetter(), r);
        }else{
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
        }
        r++;

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            r++;
        }

        //process 4 letters per iteration
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

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3);
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
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepSingleTileOtherDiagonal(subjectLetters.getCurrentLetter(), r);
            r++;
        }

        return r;
    }

    template<class Group, class State, class SubjectLettersData, class LastColumn, class TempHandler>
    __device__ __forceinline__
    void processFirstTile(
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        int subjectLength,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);
        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
        int r = 1;
        NoLastColumn noLastColumn;
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }   
        state.stepFirstTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);

        subjectLetters.shuffleCurrentLetter();
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
        state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, noLastColumn);

        subjectLetters.shuffleCurrentLetter();
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
        state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, noLastColumn);

        r += 3;

        for(; r < (group.size()) - 3; r += 4){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, noLastColumn);

            subjectLetters.shuffleCurrentLetter(); 
            //get next 4 letters from neighbor
            subjectLetters.shuffle4Letters(); 

            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, noLastColumn);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, noLastColumn);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, noLastColumn);
        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);

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
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, lastColumn);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, lastColumn);
        }
        if(r+1 < numRows){
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
        }
        if(r+2 < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            state.stepFirstTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, lastColumn);
        }

        const int totalChunksOfFour = (subjectLength) / 4;
        const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
        const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + subjectLength % 4;
        // if(threadIdx.x == 0){
        //     printf("totalChunksOfFour %d, unsavedChunksOfFour %d, numThreadsWithValidTileLastColumn %d\n", 
        //             totalChunksOfFour, unsavedChunksOfFour, numThreadsWithValidTileLastColumn);
        // }
        if(numThreadsWithValidTileLastColumn > 0){
            const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
            if(group.thread_rank() >= firstValidThread){
                tempHandler.finalSave(lastColumn, firstValidThread);
            }
        }
    }


    template<class Group, class State, class SubjectLettersData, class LeftBorder, class LastColumn, class TempHandler>
    __device__ __forceinline__
    void processIntermediateTile(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int numRows,
        int subjectLength,
        LeftBorder& leftBorder,
        LastColumn& lastColumn,
        TempHandler& tempHandler
    ){
        static_assert(group.size() >= 4);

        //process first (group.size()-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row

        int r = 1;
        NoLastColumn noLastColumn;
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
        leftBorder.shuffleDown(group);
        state.stepIntermediateTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);

        subjectLetters.shuffleCurrentLetter();
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
        leftBorder.shuffleDown(group);
        state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, noLastColumn);

        subjectLetters.shuffleCurrentLetter();
        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
        leftBorder.shuffleDown(group);
        state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, noLastColumn);

        r += 3;
        for(; r < (group.size()) - 3; r += 4){            
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, noLastColumn);

            subjectLetters.shuffleCurrentLetter(); 
            //get next 4 letters from neighbor
            subjectLetters.shuffle4Letters();
            
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder, noLastColumn);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, noLastColumn);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, noLastColumn);
        }

        for(; r < numRows - 3; r += 4){   
            // if(threadIdx.x == 0){
            //     printf("HHHHHHHHHHHHHHHH r %d numRows %d\n", r, numRows);
            // }
                
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);

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
            
            subjectLetters.shuffleCurrentLetter(); 
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder, lastColumn);
            
            if((r+4) % (group.size()) == 0){
                tempHandler.save(lastColumn);
            }  

        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }   
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder, lastColumn);
        }
        if(r+1 < numRows){
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
        }
        if(r+2 < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepIntermediateTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder, lastColumn);
        }


        const int totalChunksOfFour = (subjectLength) / 4;
        const int unsavedChunksOfFour = totalChunksOfFour % (group.size() / 4);
        const int numThreadsWithValidTileLastColumn = unsavedChunksOfFour * 4 + subjectLength % 4;
        // if(threadIdx.x == 0){
        //     printf("totalChunksOfFour %d, unsavedChunksOfFour %d, numThreadsWithValidTileLastColumn %d\n", 
        //             totalChunksOfFour, unsavedChunksOfFour, numThreadsWithValidTileLastColumn);
        // }
        if(numThreadsWithValidTileLastColumn > 0){
            const int firstValidThread = group.size() - numThreadsWithValidTileLastColumn;
            if(group.thread_rank() >= firstValidThread){
                tempHandler.finalSave(lastColumn, firstValidThread);
            }
        }
    }

    template<class Group, class State, class SubjectLettersData, class LeftBorder, class TempHandler>
    __device__ __forceinline__
    int processLastTile(
        int tileNr,
        Group& group,
        State& state,
        SubjectLettersData& subjectLetters,
        int r,
        int numRows,
        int subjectLength,
        LeftBorder& leftBorder,
        TempHandler& tempHandler
    ){
        if(r > 1){
            subjectLetters.shuffleCurrentLetter(); 
            if((r - 1) % (4*group.size()) == 0){
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
        }

        if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<0>(); }
        leftBorder.shuffleDown(group);
        if(r == 1){
            state.stepLastTileFirstDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
        }else{
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
        }
        r++;

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            r++;
        }

        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
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

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+1, tileNr, leftBorder);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+2, tileNr, leftBorder);

            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<3>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r+3, tileNr, leftBorder);            
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
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<1>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            r++;
        }
        if(r < numRows){
            subjectLetters.shuffleCurrentLetter();
            if(group.thread_rank() == 0){ subjectLetters.template setCurrentLetter<2>(); }
            leftBorder.shuffleDown(group);
            state.stepLastTileOtherDiagonal(subjectLetters.getCurrentLetter(), r, tileNr, leftBorder);
            r++;
        }

        return r;
    }


}



#endif