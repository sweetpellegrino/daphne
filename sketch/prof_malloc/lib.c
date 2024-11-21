//https://stackoverflow.com/questions/6083337/overriding-malloc-using-the-ld-preload-mechanism
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

void *(*real_malloc)(size_t) = NULL;

void *malloc(size_t size) {
    if (!real_malloc) {
        real_malloc = dlsym(RTLD_NEXT, "malloc");
    }
    void *ptr = real_malloc(size);
    //fprintf(stderr, "malloc(%zu) = %p\n", size, ptr);
    fprintf(stderr, "ALLOC:%zu\n", size);
    return ptr;
}