#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* One exact integer pass over connected-component labels. The background
 * and ineligible labels contribute nothing, matching the NumPy reference. */
#ifdef _WIN32
__declspec(dllexport)
#endif
void meteor_component_statistics(const int32_t *labels, const uint8_t *values,
                                const uint8_t *eligible, size_t pixels,
                                size_t components, uint64_t *sums, uint8_t *peaks) {
    for (size_t index = 0; index < pixels; ++index) {
        int32_t label = labels[index];
        if (label >= 0 && (size_t)label < components && eligible[label]) {
            uint8_t value = values[index];
            sums[label] += value;
            if (value > peaks[label]) peaks[label] = value;
        }
    }
}

static int32_t box(const int32_t *integral, size_t stride, int x, int y, int w, int h) {
    size_t top = (size_t)y * stride, bottom = (size_t)(y+h) * stride;
    return (integral[bottom+x+w]-integral[top+x+w])
         - (integral[bottom+x]-integral[top+x]);
}

/* A 160 -> 64 area sample spans 5 half-pixels per axis: two whole
 * pixels and one half pixel. Integer weights (4, 2, 2, 1) sum to 25.
 * No average can be a half-integer, so integer nearest rounding exactly
 * matches INTER_AREA uint8 without its per-crop coefficient allocation. */
#ifdef _WIN32
__declspec(dllexport)
#endif
void meteor_area_crop_5_2(const int32_t *integral, size_t stride, int x, int y,
                          int size, uint8_t *output) {
    for (int row=0; row<size; ++row) {
        int start_y=y+5*row/2;
        int full_y=start_y+(row&1), half_y=start_y+((row&1)?0:2);
        for (int column=0; column<size; ++column) {
            int start_x=x+5*column/2;
            int full_x=start_x+(column&1), half_x=start_x+((column&1)?0:2);
            int total=4*box(integral,stride,full_x,full_y,2,2)
                     +2*box(integral,stride,half_x,full_y,1,2)
                     +2*box(integral,stride,full_x,half_y,2,1)
                     +box(integral,stride,half_x,half_y,1,1);
            output[row*size+column]=(uint8_t)((total+12)/25);
        }
    }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void meteor_area_crop_i32(const int32_t *integral, size_t stride, int x, int y,
                          int scale_x, int scale_y, int size, uint8_t *output) {
    float scale = 1.0f / (float)(scale_x * scale_y);
    for (int row = 0; row < size; ++row) {
        size_t top = (size_t)(y + row * scale_y) * stride;
        size_t bottom = (size_t)(y + (row + 1) * scale_y) * stride;
        for (int column = 0; column < size; ++column) {
            int left = x + column * scale_x;
            int right = left + scale_x;
            int32_t total = (integral[bottom + right] - integral[top + right])
                          - (integral[bottom + left] - integral[top + left]);
            /* OpenCV's 2x2 uint8 kernel rounds ties upwards. The generic
             * integer-area kernel uses float32 scale and nearest-even. */
            output[row * size + column] = (uint8_t)(scale_x == 2 && scale_y == 2
                ? (total + 2) / 4 : lrintf((float)total * scale));
        }
    }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void meteor_area_crops(const int32_t **integrals, size_t count, size_t stride,
                       int x, int y, int width, int height, uint8_t *output) {
    for (size_t frame=0; frame<count; ++frame) {
        if (width==160 && height==160)
            meteor_area_crop_5_2(integrals[frame],stride,x,y,64,output+frame*4096);
        else
            meteor_area_crop_i32(integrals[frame],stride,x,y,width/64,height/64,64,output+frame*4096);
    }
}

/* Sort independent uint8 temporal histories with a fixed compare-exchange
 * network. Plane-contiguous comparisons vectorize without changing values.
 * Pad with 255, so both middle samples of the original history stay exact. */
#ifdef _WIN32
__declspec(dllexport)
#endif
int meteor_temporal_median_range(const uint8_t *input, size_t frames, size_t stride,
                                 size_t offset, size_t pixels, double *output) {
    size_t count=1;
    while (count<frames) count*=2;
    uint8_t *work=malloc(count*pixels);
    if (!work) return 0;
    for (size_t frame=0;frame<frames;++frame)
        memcpy(work+frame*pixels,input+frame*stride+offset,pixels);
    memset(work+frames*pixels,255,(count-frames)*pixels);
    for (size_t k=2;k<=count;k*=2) {
        for (size_t j=k/2;j;j/=2) {
            for (size_t i=0;i<count;++i) {
                size_t other=i^j;
                if (other<=i) continue;
                uint8_t *left=work+i*pixels, *right=work+other*pixels;
                if (!(i&k)) {
                    for (size_t p=0;p<pixels;++p) {
                        uint8_t a=left[p],b=right[p];
                        left[p]=a<b?a:b; right[p]=a>b?a:b;
                    }
                } else {
                    for (size_t p=0;p<pixels;++p) {
                        uint8_t a=left[p],b=right[p];
                        left[p]=a>b?a:b; right[p]=a<b?a:b;
                    }
                }
            }
        }
    }
    const uint8_t *low=work+((frames-1)/2)*pixels,*high=work+(frames/2)*pixels;
    for (size_t p=0;p<pixels;++p) output[p]=((double)low[p]+high[p])*.5;
    free(work);
    return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int meteor_temporal_median(const uint8_t *input, size_t frames, size_t pixels, double *output) {
    return meteor_temporal_median_range(input,frames,pixels,0,pixels,output);
}

/* MAD histories use exact signed 16-bit differences. The same independent
 * sorting network preserves NumPy's median, including negative values and
 * the double-precision mean of the two middle samples. */
#ifdef _WIN32
__declspec(dllexport)
#endif
int meteor_temporal_median_i16(const int16_t *input, size_t frames, size_t pixels, double *output) {
    if (!frames || frames>32) return 0;
    size_t count=1;
    while (count<frames) count*=2;
    int16_t *work=malloc(count*pixels*sizeof(int16_t));
    if (!work) return 0;
    memcpy(work,input,frames*pixels*sizeof(int16_t));
    for (size_t i=frames*pixels;i<count*pixels;++i) work[i]=INT16_MAX;
    for (size_t k=2;k<=count;k*=2) {
        for (size_t j=k/2;j;j/=2) {
            for (size_t i=0;i<count;++i) {
                size_t other=i^j;
                if (other<=i) continue;
                int16_t *left=work+i*pixels,*right=work+other*pixels;
                if (!(i&k)) {
                    for (size_t p=0;p<pixels;++p) {
                        int16_t a=left[p],b=right[p];
                        left[p]=a<b?a:b; right[p]=a>b?a:b;
                    }
                } else {
                    for (size_t p=0;p<pixels;++p) {
                        int16_t a=left[p],b=right[p];
                        left[p]=a>b?a:b; right[p]=a<b?a:b;
                    }
                }
            }
        }
    }
    const int16_t *low=work+((frames-1)/2)*pixels,*high=work+(frames/2)*pixels;
    for (size_t p=0;p<pixels;++p) output[p]=((double)low[p]+high[p])*.5;
    free(work);
    return 1;
}
