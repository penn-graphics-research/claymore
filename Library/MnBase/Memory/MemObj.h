#ifndef __MEM_OBJ_H_
#define __MEM_OBJ_H_

namespace mn {

    struct MemBackend {
        using uchar = unsigned char;
        enum class physical : uchar {
            cpu = 0,
            gpu = 1
        };
    };

    struct MemResource {
        union {
			void* ptr;
            uintptr_t ptrval;
			uint64_t offset;    ///< only legal for 64-bit app
		};
    };

    struct MemStack
    {
        char *top;
        char *end;
    };

    struct MemBlock {
        void *mem;
        std::size_t size;
    };

} // namespace mn

#endif