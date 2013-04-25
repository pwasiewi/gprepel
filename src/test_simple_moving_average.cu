////////////////////////////////////////////////////////////////////////////////////////////////////
//  gpRepel : An R package for GPU computing - testing the moving average function
//  COMPILE: nvcc -arch sm_11 test_simple_moving_average.cu
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
//
//  Author: Piotr Wąsiewicz pwasiewi@gmail.com
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpRepel.h"


using namespace std;

typedef thrust::tuple<Numeric,Numeric> 								Numeric2;
typedef typename thrust::device_vector<Numeric>::iterator         	NumericIterator;
typedef typename thrust::tuple<NumericIterator, NumericIterator>  	NumericIteratorTuple;
typedef typename thrust::zip_iterator<NumericIteratorTuple>       	Numeric2Iterator;
typedef thrust::tuple<Numeric,Numeric,Numeric> 						Numeric3;
typedef typename thrust::tuple<NumericIterator, NumericIterator, NumericIterator>  NumericIteratorTuple3;
typedef typename thrust::zip_iterator<NumericIteratorTuple3>       	Numeric3Iterator;

////////////////////////////////////////////////////////////////////////////////////////////////////
// VecReorder - the gpu functor implementing the dot product between 3d vectors
////////////////////////////////////////////////////////////////////////////////////////////////////
struct VecReorder : public thrust::binary_function<Numeric2,Numeric2,Numeric>
{
    Numeric w, maxb;
    Numeric ai,bi,av,bv, result;

    __host__ __device__
    VecReorder(Numeric w, Numeric maxb) : w(w), maxb(maxb) {}
    __host__ __device__
        Numeric operator()(const Numeric2& a, const Numeric2& b) const
        {
            Numeric ai=(int) thrust::get<0>(a) % (int) maxb;
            Numeric av=thrust::get<1>(a);
            Numeric bi=(int) thrust::get<0>(b) % (int) maxb;
            Numeric bv=thrust::get<1>(b);
	    	int lastone = (int) maxb*((int) thrust::get<0>(a) / (int) maxb)-1;
	    	if(ai > bi)
            	return thrust::get<0>(a);
	    	else
	    		return lastone;
        }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// minus_and_divide_zip - gpu functor implementing the dot product between 3d vectors
////////////////////////////////////////////////////////////////////////////////////////////////////
struct  minus_and_divide_zip : public thrust::binary_function<Numeric3,Numeric3,Numeric>
{
    Numeric w, maxb;
    Numeric ai,bi,av,bv, result;

    __host__ __device__
    minus_and_divide_zip(Numeric w, Numeric maxb) : w(w), maxb(maxb) {}
    __host__ __device__
    Numeric operator()(const Numeric3& a, const Numeric3& b) const
    {
		Numeric ai=(int) thrust::get<0>(a) % (int) maxb;
		Numeric av=thrust::get<1>(a);
		Numeric bi=(int) thrust::get<0>(b) % (int) maxb;
		Numeric bv=thrust::get<1>(b);
	    int lastone = (int) maxb*((int) thrust::get<0>(a) / (int) maxb);
	    if(ai > bi)
                return (av - bv)/w;
	    else
	    	if((int)w - 1 != (int) ai)
	    		return (thrust::get<2>(b) - thrust::get<1>(b)) / ((int)w - (int)ai -1);
	    	else
	    		return 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// simple_moving_average - GPU function of the simple average with a window w points forward, 
// after a given point; idata, vout - input and output matrices with m (rows) x n (cols) dimensions,
// 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename InputVector, typename OutputVector>
void simple_moving_average(size_t m, size_t n, const InputVector& idata, size_t w, OutputVector& vout)
{
    typedef typename InputVector::value_type T;

    if (idata.size() < w)
        return;
    thrust::device_vector<size_t> output(m*n);
    thrust::device_vector<Numeric> voutput(m*n);
    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end());

    thrust::inclusive_scan(idata.begin(), idata.end(), data.begin());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin(), data.begin()));
    Numeric2Iterator firstw = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin() + w, data.begin() + w));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(vindex.end(),   data.end()));

    thrust::transform(firstw, last, first, output.begin(), VecReorder(w,m));
 
    thrust::gather(output.begin(), output.end(), data.begin(), voutput.begin());

    Numeric3Iterator first3 = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin(), data.begin(), voutput.begin()));
    Numeric3Iterator firstw3 = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin() + w, data.begin() + w, voutput.begin() + w));
    Numeric3Iterator last3  = thrust::make_zip_iterator(thrust::make_tuple(vindex.end(), data.end(), voutput.end()));

    thrust::transform(firstw3, last3, first3, vout.begin(), minus_and_divide_zip(w,m));
    thrust::fill(vout.end()-w,vout.end(),vout[vout.size()-w-1]);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// print an array m x n with vectors in columns
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
            std::cout << " " << h_data[j + i * m] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// print an array m x n with vectors in columns as one vector: one after another
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void printvec(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < m*n; i++)
        std::cout << setw(4) << h_data[i] << " ";
    std::cout << "\n";
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Test of simple_moving_average
////////////////////////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  const int NDATA=20;
  int values[NDATA] = {	3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6};
  int w = 3, m = 10, n = 2;
   
  // transfer to device and compute sum
  thrust::device_vector<int> idata(values, values + NDATA);
  thrust::device_vector<Numeric> vout(m*n);
 
  cout << "moving average from i+1 to i+w" << endl;

  cout << "idata (two vectors one after another):" << endl;
  print(m,n,idata);

  if (idata.size() < w)
     return 1;
  thrust::device_vector<size_t> output(m*n);
  thrust::device_vector<Numeric> voutput(m*n);
  thrust::device_vector<Numeric> data(m*n);
  thrust::device_vector<Numeric> vindex(m*n);
  thrust::sequence(vindex.begin(),vindex.end());

  cout << "vindex (index of all joint vectors):" << endl;
  printvec(1,NDATA,vindex);

  thrust::inclusive_scan(idata.begin(), idata.end(), data.begin());
  cout << "data (incrementing sum with a given point included):" << endl;
  printvec(1,NDATA,data);
  Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin(), data.begin()));
  Numeric2Iterator firstw = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin() + w, data.begin() + w));
  Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(vindex.end(),   data.end()));

  thrust::transform(firstw, last, first, output.begin(), VecReorder(w,m));
  cout << "output (w points left shifted index):" << endl;
  printvec(1,NDATA,output); 
  thrust::gather(output.begin(), output.end(), data.begin(), voutput.begin());
  cout << "voutput (w points left shifted vectors, each separately):" << endl;
  printvec(1,NDATA,voutput); 
  Numeric3Iterator first3 = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin(), data.begin(), voutput.begin()));
  Numeric3Iterator firstw3 = thrust::make_zip_iterator(thrust::make_tuple(vindex.begin() + w, data.begin() + w, voutput.begin() + w));
  Numeric3Iterator last3  = thrust::make_zip_iterator(thrust::make_tuple(vindex.end(), data.end(), voutput.end()));

  thrust::transform(firstw3, last3, first3, vout.begin(), minus_and_divide_zip(w,m));
  cout << "vout (moving average from point + 1 to w):" << endl;
  printvec(1,NDATA,vout); 
  thrust::fill(vout.end()-w,vout.end(),vout[vout.size()-w-1]);
  cout << "vout after fill ():" << endl;
  printvec(1,NDATA,vout); 
  return 0;
}
