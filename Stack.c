#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "mpi.h"

#define opsCount 10000

typedef struct {
	uint64_t rank : 11;
	uint64_t offset : 53;
} nodePtr;

typedef struct {
	int val;
	nodePtr next;
} node;

typedef struct{
    nodePtr dummy;
	nodePtr head;
} Stack;

static node** allocNodes = NULL; 
static node** allocNodesTmp = NULL; 
static int allocNodeSize = 0; 
static int allocNodeCount = 0; 
static const nodePtr nullPtr = {2047, (MPI_Aint)MPI_BOTTOM};

int succPush = 0;
int succPop = 0;
int totalElementCount = 0;

uint64_t allocElem(int val, MPI_Win win) {
	MPI_Aint disp;
	node* allocNode;

	MPI_Alloc_mem(sizeof(node), MPI_INFO_NULL, &allocNode);

	allocNode->val = val;
	allocNode->next = nullPtr;

	MPI_Win_attach(win, allocNode, sizeof(node));

	if (allocNodeCount == allocNodeSize) {
		allocNodeSize += 100;
		allocNodesTmp = (node**)realloc(allocNodes, allocNodeSize * sizeof(node*));
		if (allocNodesTmp != NULL)
			allocNodes = allocNodesTmp;
		else {
			printf("Error while allocating memory!\n");
			return 0;
		}
	}
	
	allocNodes[allocNodeCount] = allocNode;
	allocNodeCount++;
	MPI_Get_address(allocNode, &disp);

	return disp;
}

int readVal(nodePtr ptr, MPI_Win win)
{
	int result = 0;
	MPI_Get((void*)&result, 1, MPI_INT, ptr.rank, ptr.offset + offsetof(node, val),
		1, MPI_INT, win);
	MPI_Win_flush(ptr.rank, win);
	printf("Val %d\n", result);
}

nodePtr getHead(Stack s, MPI_Win win)
{
    nodePtr result = {0};

    MPI_Fetch_and_op(NULL, (void*)&result, MPI_LONG_LONG, s.dummy.rank,
        s.dummy.offset + offsetof(node, next), MPI_NO_OP, win);
    MPI_Win_flush(s.dummy.rank, win);

    return result;
}

void changeNext(nodePtr oldHead, nodePtr newHead, MPI_Win win){
    
    nodePtr result = { 0 };

    MPI_Fetch_and_op((void*)&oldHead, (void*)&result, MPI_LONG_LONG, newHead.rank,
        newHead.offset + offsetof(node, next), MPI_REPLACE, win);
    MPI_Win_flush(newHead.rank, win);
}

void push(int val, int rank, Stack s, MPI_Win win)
{
    nodePtr curHead = {0}, newHead = {0}, result = {0};

    newHead.rank = rank;
    newHead.offset = allocElem(val, win);
    
    while(1){
        curHead = getHead(s, win);

        changeNext(curHead, newHead, win);

        MPI_Compare_and_swap((void*)&newHead, (void*)&curHead, (void*)&result, MPI_LONG_LONG, 
            s.dummy.rank, s.dummy.offset + offsetof(node, next), win);
        MPI_Win_flush(s.dummy.rank, win);

        if(result.rank == curHead.rank && result.offset == curHead.offset) {
			succPush++;
			return;
		}
    }
}

nodePtr getNextHead(nodePtr head, MPI_Win win)
{
    nodePtr result = {0};

    MPI_Fetch_and_op(NULL, (void*)&result, MPI_LONG_LONG, head.rank,
        head.offset + offsetof(node, next), MPI_NO_OP, win);
    MPI_Win_flush(head.rank, win);

    return result;
}

void pop(Stack s, MPI_Win win)
{
    nodePtr curHead = {0}, result = {0}, nextHead = {0};

    while(1){
        curHead = getHead(s, win);

		if(curHead.rank == nullPtr.rank) return;

        nextHead = getNextHead(curHead, win);

        MPI_Compare_and_swap((void*)&nextHead, (void*)&curHead, (void*)&result, MPI_LONG_LONG, 
            s.dummy.rank, s.dummy.offset + offsetof(node, next), win);
        MPI_Win_flush(s.dummy.rank, win);

        if(result.rank == curHead.rank && result.offset == curHead.offset) {
			succPop++;
			return;
		}
    }
}

void printStack(int rank, Stack Stack, MPI_Win win)
{
	node curNode = {0};
	nodePtr curNodePtr = getNextHead(Stack.dummy, win);
	int i = 0;
	
	printf("Rank[%d]: Result Stack is: \n", rank);

	while (curNodePtr.offset != nullPtr.offset && curNodePtr.rank != nullPtr.rank) {

		MPI_Get((void*)&curNode, sizeof(node), MPI_BYTE,
			curNodePtr.rank, curNodePtr.offset, sizeof(node), MPI_BYTE, win);
		MPI_Win_flush(curNodePtr.rank, win);
		
		curNodePtr = curNode.next;
		if(curNodePtr.rank != nullPtr.rank){
			printf("------val %d was inserted by rank %d at displacement %x next rank %d next displacement %x------"
			"\n", curNode.val, curNodePtr.rank, curNodePtr.offset, curNode.next.rank, curNode.next.offset);
			i++;
		}
	}
	totalElementCount = i;
}

void init(int argc, char* argv[], int* procid, int* numproc, MPI_Win* win)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, procid);
	MPI_Comm_size(MPI_COMM_WORLD, numproc);
	MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win);
}

void showStat(int procid, int numproc, double elapsedTime, int ops)
{
	int results[2] = { 0 };
	int elemCount = 0;

	if(procid != 0) {
		results[0] = succPush; results[1] = succPop;
		MPI_Send((void*)&results, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
	} else {
		elemCount += (succPush - succPop);
		for(int i = 1; i < numproc; i++){
			MPI_Recv((void*)&results, 2, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			elemCount += (results[0] - results[1]);
		}
	}
	if(procid == 0) {
		printf("Total element count = %d\n", totalElementCount);
		printf("Expected element count = %d\n", elemCount);
		printf("Test result: total elapsed time = %f ops/sec = %f\n", elapsedTime, ops/elapsedTime);
		totalElementCount == elemCount ? puts("Stack Integrity: True") : puts("Stack Integrity: False");
	}
}

void runTest(Stack stack, MPI_Win win, int rank, int numproc, int testSize)
{
	double startTime, endTime, elapsedTime;
	int len;
	char procName[MPI_MAX_PROCESSOR_NAME];

	srand(time(0) + rank);

	MPI_Win_lock_all(0, win);
	MPI_Barrier(MPI_COMM_WORLD);
	startTime = MPI_Wtime();

	for (int i = 0; i < testSize; i++) {
		if(rand() % 2 == 0) push(i, rank, stack, win);
        else pop(stack, win);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	endTime = MPI_Wtime();

	elapsedTime = endTime - startTime;

	if(rank == 0) printStack(rank, stack, win);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Win_unlock_all(win);	

	showStat(rank, numproc, elapsedTime, numproc * testSize);
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Get_processor_name(procName, &len);
    	printf("rank %d of all %d ranks was launched at %s\n", rank, numproc, procName);

	for (int i = 0; i < allocNodeCount; i++) {
		MPI_Win_detach(win, allocNodes[i]);
		MPI_Free_mem(allocNodes[i]);
	}

	MPI_Win_free(&win);
	MPI_Finalize();
}

void testInit(int argc, char* argv[], int testSize)
{
	int rank, numproc, elemCount = 0;
	MPI_Win win;
	nodePtr head = { 0 }, dummy = { 0 };
	Stack stack = { 0 };

	init(argc, argv, &rank, &numproc, &win);
	if (rank == 0) {
		head.offset = allocElem(-1, win); head.rank = 0;
		dummy.offset = allocElem(-1, win); dummy.rank = 0;
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		MPI_Put((void*)&head, sizeof(nodePtr), MPI_BYTE, 0, dummy.offset + offsetof(node, next),
			sizeof(nodePtr), MPI_BYTE, win);
		MPI_Win_unlock(0, win);
		stack.dummy = dummy;
		stack.head = head;
	}

	MPI_Bcast(&head, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dummy, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

	stack.dummy = dummy;
	stack.head = head;

	runTest(stack, win, rank, numproc, testSize);
}

int main(int argc, char* argv[])
{
	testInit(argc, argv, opsCount);
	return 0;
}
