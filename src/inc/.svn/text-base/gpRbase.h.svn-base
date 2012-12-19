////////////////////////////////////////////////////////////////////////
//  gpRepel : An R packge for GPU computing
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; version 3 of the License.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#ifndef __gpRBASE_H__
#define __gpRBASE_H__


#include <stdexcept>
#include <math_constants.h>


#ifdef __cplusplus
extern "C"
{
#endif 

#ifdef _gpRepel_

#define gpR_PROBLEM_BUFSIZE       R_PROBLEM_BUFSIZE
#define gpRPrintf                 Rprintf

#else

#define gpR_PROBLEM_BUFSIZE       4096
#define gpRPrintf                 printf

#endif


#ifdef __cplusplus
}
#endif 


namespace gpR {


////////////////////////////////////////////////////////////////////////
// RuntimeException
//
class RuntimeException : public std::runtime_error {

public:
    RuntimeException(const char* msg, const char* file, const int line) :
        std::runtime_error("") {
        
        sprintf(m_buf, "%s(%i): %s\n", file, line, msg);
    }
    
    RuntimeException(const char* method, const char* msg, 
            cudaError err, const char* file, const int line) :
        std::runtime_error("") {
     
        sprintf(m_buf, "%s(%i): %s: %s - %s.\n",
            file, line, method, msg, cudaGetErrorString(err));
    }
     
    virtual const char* what() const throw() {
        return m_buf;
    }

private:
    char m_buf[gpR_PROBLEM_BUFSIZE];
};


class AppException : public std::exception {

public:
    AppException(char* msg) {
        strncpy(m_buf, msg, gpR_PROBLEM_BUFSIZE-1);
    }
    
    virtual const char* what() const throw() {
        return m_buf;
    }

private:
    char m_buf[gpR_PROBLEM_BUFSIZE];
};


}   // namespace gpR



////////////////////////////////////////////////////////////////////////
// numeric constants
//

#define gpR_NON_NA(a)           (!isnan(a))
#define gpR_BOTH_NON_NA(a, b)   (!isnan(a) && !isnan(b))

#define gpR_FINITE(a)           (!isinf(a))
#define gpR_BOTH_FINITE(a, b)   (!isinf(a) && !isinf(b))

// only single precision for now
#define gpR_NUMERIC_INF         CUDART_INF_F
#define gpR_NUMERIC_NAN         CUDART_NAN_F
#define gpR_NUMERIC_MAX         CUDART_MAX_NORMAL_F
#define gpR_NUMERIC_MIN_DENORM  CUDART_MIN_DENORM_F

#define NUMERIC_MAX		        FLT_MAX
#define NUMERIC_MIN		        FLT_MIN


////////////////////////////////////////////////////////////////////////
// debug macros
//

#ifdef _DEBUG

#define gpR_MSG(x)              gpRPrintf("%s\n", x)
#define gpR_MSG2(fmt, x)        gpRPrintf(fmt, x)

#else

#define gpR_MSG(x)              ((void)0)
#define gpR_MSG2(fmt, x)        ((void)0)

#endif     // _DEBUG


////////////////////////////////////////////////////////////////////////
// error handling
//

#define gpRError(msg)           __gpRError((msg), __FILE__, __LINE__)
#define gpRSafeCall(err)        __gpRSafeCall((err), __FILE__, __LINE__)
#define gpRCheckMsg(msg)        __gpRCheckMsg((msg), __FILE__, __LINE__)


inline void __gpRError(const char* errorMessage, const char* file, const int line)
{
    throw new gpR::RuntimeException(errorMessage, file, line);
}

inline void __gpRSafeCall(cudaError err, const char* file, const int line) {

    if( cudaSuccess != err) {
        throw gpR::RuntimeException("gpRSafeCall", 
                        "CUDA runtime API error", err, file, line);
    }
}

inline void __gpRCheckMsg(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        throw gpR::RuntimeException("gpRCheckMsg", 
                        errorMessage, err, file, line);
    }
}



#endif     // __gpRBASE_H__



