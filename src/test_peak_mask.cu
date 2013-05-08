////////////////////////////////////////////////////////////////////////////////////////////////////
//  gpRepel : An R package for GPU computing - testing the moving average function
//  COMPILE: nvcc -arch sm_11 test_*.cu
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
// mask01 - the gpu functor of making mask, where values greater than w are set to 1
//
template <typename T>
struct mask01 : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    mask01(T w) : w(w) {}

	__host__ __device__
    T operator()(const T& a) const
    {
    	if(a > w)
    		return 1;
    	else
    		return 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// segmented by peaks sums 
// 
template <typename InputVector, typename OutputVector>
void segmented_peak_sums(size_t m, size_t n, const InputVector& idata, OutputVector& data)
{
  typedef typename InputVector::value_type T;

  thrust::device_vector<Numeric> peakmask(m*n);

  thrust::transform(idata.begin(), idata.end(), peakmask.begin(), mask01<Numeric>(Integer(0)));
  //cout << "peakmask after fill ():" << endl;
  //printvec(1,NDATA,peakmask); 
  
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), idata.begin(),data.begin());
  //cout << "segmented by peakmask sums:" << endl;
  //printvec(1,NDATA,data);
   
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  
  thrust::reverse(data.begin(), data.end());
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// Test of peak mask
////////////////////////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  const int NDATA=20;
  int values[NDATA] = {	0, 0, 2, 3, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 0};
  int m = 10, n = 2;
   
  // transfer to device
  thrust::device_vector<int> 	 idata(values, values + NDATA);
  cout << "testing peak mask and integral" << endl;
  cout << "idata (two vectors one after another):" << endl;
  printvec(m,n,idata);

  thrust::device_vector<Numeric> data(m*n);
  
  /*
  thrust::device_vector<Numeric> peakmask(m*n);

  thrust::transform(idata.begin(), idata.end(), peakmask.begin(), mask01<Numeric>(Integer(0)));
  cout << "peakmask after fill ():" << endl;
  printvec(1,NDATA,peakmask); 
  
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), idata.begin(),data.begin());
  cout << "segmented by peakmask sums:" << endl;
  printvec(1,NDATA,data);
   
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  
  thrust::reverse(data.begin(), data.end());
  */
  segmented_peak_sums(m, n, idata, data);
  cout << "data after max(sum) in peakmask segments:" << endl;
  printvec(1,NDATA,data); 
  return 0;
}
