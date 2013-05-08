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

const int NDATA=20;
const int NAVG=1;

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
// plus_and_divide - a gpu functor of two arguments added and divided by a constant
//
template <typename T>
struct plus_and_divide : public thrust::binary_function<T,T,T>
{
    T w;

    __host__ __device__
    plus_and_divide(T w) : w(w) {}

    __host__ __device__
    T operator()(const T& a, const T& b) const
    {
        return (a + b) / w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// double_moving_average - gpu function for the gpu average of two moving averages with windows 
// after and before the given point, joint of two simple_moving_average (one on a reversed copy)
template <typename InputVector, typename OutputVector>
void double_moving_average(size_t m, size_t n, const InputVector& igva, size_t w, OutputVector& gvd)
{
    typedef typename InputVector::value_type T;
    if (igva.size() < w)
        return;

    thrust::device_vector<T> gva(igva.size());
    thrust::device_vector<T> gvb(igva.size());
    thrust::device_vector<T> gvc(igva.size());
    thrust::copy(igva.begin(), igva.end(), gva.begin());

    simple_moving_average(m,n,gva, w, gvb);
    thrust::reverse(gva.begin(), gva.end());
    simple_moving_average(m,n,gva, w, gvc);
    thrust::reverse(gvc.begin(), gvc.end());
    thrust::reverse(gva.begin(), gva.end());
    thrust::transform(gvc.begin(), gvc.end(), gvb.begin(), gvd.begin(), plus_and_divide<T>(T(2)));
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
        std::cout << fixed << setw(5) << setprecision(2) << h_data[i] << " ";
    std::cout << "\n";
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// oneup - the gpu functor of constant value substraction, where negative values are set to 0
//
template <typename T>
struct oneup : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    oneup(T w) : w(w) {}

	__host__ __device__
    T operator()(const T& a) const
    {
    	if(a <= w)
    		return 0;
    	else
    		return a-w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// onedown - the gpu functor from constant value the wave substraction, 
// where negative values are set to 0
//
template <typename T>
struct onedown : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    onedown(T w) : w(w) {}

    __host__ __device__
    T operator()(const T& a) const
    {
    	if(a >= w)
    		return 0;
    	else
    		return w-a;
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
// zipup - the gpu functor 
//
struct  zipup : public thrust::unary_function<Numeric2,Numeric>
{
     Numeric va,da;

    __host__ __device__
    Numeric operator()(const Numeric2& a) const
    {
		Numeric va=thrust::get<0>(a);
		Numeric da=thrust::get<1>(a);
    	if(va <= da)
    		return 0;
    	else
    		return va-da;
     }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// compare_zip - the gpu functor 
//
struct  compare_zip : public thrust::binary_function<Numeric2,Numeric2,Numeric>
{
     Numeric va,vb, da, db;

    __host__ __device__
    Numeric operator()(const Numeric2& a, const Numeric2& b) const
    {
		Numeric va=thrust::get<0>(a);
		Numeric da=thrust::get<1>(a);
		Numeric vb=thrust::get<0>(b);
		Numeric db=thrust::get<1>(b);
	    if(va > 0){
	    	if(da > 0 && db <0){
	    		return 1;
	    	}
	    	else
	    		return 0;
	    }
	    else
	    	return 0;
     }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// kindcreate - the gpu functor of integer division by a constant m
//
struct kindcreate : public thrust::unary_function<Numeric,Numeric>
{
    Integer m, n;

    __host__ __device__
    kindcreate(Integer m, Integer n) : m(m), n(n) {}
    __host__ __device__
    Numeric operator()(const Numeric& a) const
    {
        //Integer ai=(int) ((int) a) % m;
        Integer av=(int) ((int) a) / m;
        //if(a > 0)
        return (int) av;
   }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
// avg find peaks 
// 
template <typename InputVector, typename OutputVector>
void avg_find_peaks(size_t m, size_t n, const Numeric& w1, const Numeric& w2, const Numeric& w3, 
InputVector& orgbasoff, InputVector& avgbasoff,OutputVector& out)
{
  typedef typename InputVector::value_type T;
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;

  thrust::device_vector<Numeric> data(m*n);
  thrust::device_vector<Numeric> data2(m*n);
  
  //the same number within each vector
  thrust::device_vector<Integer> vindex(m*n);
  thrust::sequence(vindex.begin(),vindex.end(),0);
  thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));
  
  thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoff.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
  thrust::reverse(data.begin(), data.end());
  thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
  thrust::reverse(data2.begin(), data2.end());
  thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
  thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());
  
  	//cout << "peaks divided by average:" << endl;
  	//printvec(1,NDATA,data2);
  	 
  Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoff.begin(), data2.begin()));
  Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoff.end(),   data2.end()));

  thrust::transform(first, last, data.begin(), zipup());

  	//cout << "peaks zipup:" << endl;
  	//printvec(1,NDATA,data);

  thrust::transform(data.begin()+1, data.end(), data.begin(), out.begin(), thrust::minus<Numeric>());
  double_moving_average(m,n,out, w3, data2);

  Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
  Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
  Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

  thrust::transform(first0, last0, first1, out.begin(), compare_zip());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// segregate peaks up and down
// 
template <typename InputVector, typename OutputVector>
void create_peaks(InputVector& data, InputVector& base, OutputVector& wavbasoffp, OutputVector& wavbasoffd)
{
	typedef typename InputVector::value_type T;
	thrust::device_vector<Numeric> data2(data.begin(),data.end());
   	thrust::transform(data.begin(), data.end(), base.begin(), data2.begin(), thrust::divides<Numeric>());
  	thrust::transform(data2.begin(), data2.end(), wavbasoffp.begin(), oneup<Numeric>(Numeric(1)));
  	thrust::transform(data2.begin(), data2.end(), wavbasoffd.begin(), onedown<Numeric>(Numeric(1)));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// find_doublemaxpeaks - a gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_doublemaxpeaks(size_t m, size_t n, const InputVector& idata, const Numeric& w1, const Numeric& w2, const Numeric& w3, OutputVector& pout, OutputVector& avgbasoffp, OutputVector& dout, OutputVector& avgbasoffd)
{
	thrust::device_vector<Numeric> orgbasoffp(m*n);
	thrust::device_vector<Numeric> orgbasoffd(m*n);

    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> data2(m*n);

    double_moving_average(m,n,idata, w2, data2);
    thrust::transform(idata.begin(), idata.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());

    //original peaks up and down
    thrust::transform(data.begin(), data.end(), orgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), orgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
 
    double_moving_average(m,n,idata, w1, data);
    thrust::transform(data.begin(), data.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());
	//average peaks up and down
    thrust::transform(data.begin(), data.end(), avgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), avgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

	//the same number within each vector
    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));

    //orgbasoffp, avgbasoffp
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffp.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.begin(), data2.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), pout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,pout, w3, data2);

    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

    thrust::transform(first0, last0, first1, pout.begin(), compare_zip());

    //orgbasoffd, avgbasoffd
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffd.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.begin(), data2.begin()));
    last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), dout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,dout, w3, data2);

    first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));
    
    thrust::transform(first0, last0, first1, dout.begin(), compare_zip());
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
// tone - the gpu functor of making 1, where values are equal to 1
//
template <typename T>
struct tone : public thrust::unary_function<T,T>
{
    T w;
	__host__ __device__
    tone(T w) : w(w) {}

	__host__ __device__
    T operator()(const T& a) const
    {
    	if(a > w)
    		return a;
    	else
    		return 1;
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
  thrust::device_vector<Numeric> peakbase(peakmask.begin(), peakmask.end());
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), peakbase.begin(),peakbase.begin());
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), idata.begin(),data.begin());
  thrust::reverse(peakbase.begin(), peakbase.end());
  thrust::reverse(data.begin(), data.end());
  thrust::reverse(peakmask.begin(), peakmask.end());
  thrust::equal_to<Numeric> binary_pred;
  thrust::maximum<Numeric>  binary_max;
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), data.begin(), data.begin(),binary_pred,binary_max);
  thrust::inclusive_scan_by_key(peakmask.begin(), peakmask.end(), peakbase.begin(), peakbase.begin(),binary_pred,binary_max);
  thrust::reverse(peakbase.begin(), peakbase.end());
  thrust::reverse(data.begin(), data.end());
  thrust::transform(peakbase.begin(), peakbase.end(), peakbase.begin(), tone<Numeric>(Integer(0)));
  //cout << "data of peakbase:" << endl;
  //printvec(1,NDATA,peakbase); 
  thrust::transform(data.begin(), data.end(), peakbase.begin(), data.begin(), thrust::divides<Numeric>());

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// find_doublemaxpeaks_opt - a gpu function for 
//
template <typename InputVector, typename OutputVector>
void find_doublemaxpeaks_opt(size_t m, size_t n, const InputVector& idata, const Numeric& w1, const Numeric& w2, const Numeric& w3, 
OutputVector& pout, OutputVector& avgbasoffp, OutputVector& pintegralp, OutputVector& dout, OutputVector& avgbasoffd, OutputVector& pintegrald)
{
	thrust::device_vector<Numeric> orgbasoffp(m*n);
	thrust::device_vector<Numeric> orgbasoffd(m*n);

    thrust::device_vector<Numeric> data(m*n);
    thrust::device_vector<Numeric> data2(m*n);

    double_moving_average(m,n,idata, w2, data2);
    thrust::transform(idata.begin(), idata.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());

    //original peaks up and down
    thrust::transform(data.begin(), data.end(), orgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), orgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

    thrust::equal_to<Numeric> binary_pred;
    thrust::maximum<Numeric>  binary_max;
 
    double_moving_average(m,n,idata, w1, data);
    thrust::transform(data.begin(), data.end(), data2.begin(), data.begin(), thrust::divides<Numeric>());
	//average peaks up and down
    thrust::transform(data.begin(), data.end(), avgbasoffp.begin(), oneup<Numeric>(Numeric(1)));
    thrust::transform(data.begin(), data.end(), avgbasoffd.begin(), onedown<Numeric>(Numeric(1)));

	//the same number within each vector
    thrust::device_vector<Integer> vindex(m*n);
    thrust::sequence(vindex.begin(),vindex.end(),0);
    thrust::transform(vindex.begin(), vindex.end(), vindex.begin(), kindcreate(Integer(m),Integer(n)));

    //orgbasoffp, avgbasoffp
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffp.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    Numeric2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.begin(), data2.begin()));
    Numeric2Iterator last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffp.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), pout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,pout, w3, data2);

    Numeric2Iterator first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    Numeric2Iterator first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    Numeric2Iterator last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));

    thrust::transform(first0, last0, first1, pout.begin(), compare_zip());

    //orgbasoffd, avgbasoffd
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), orgbasoffd.begin(), data.begin(), binary_pred,thrust::plus<Numeric>());
    thrust::reverse(data.begin(), data.end());
    thrust::inclusive_scan_by_key(vindex.begin(), vindex.end(), data.begin(), data2.begin(),binary_pred,binary_max);
    thrust::reverse(data2.begin(), data2.end());
    thrust::fill(data.begin(),data.end(),m/(NAVG/w1));
    thrust::transform(data2.begin(), data2.end(), data.begin(), data2.begin(), thrust::divides<Numeric>());

    first = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.begin(), data2.begin()));
    last  = thrust::make_zip_iterator(thrust::make_tuple(avgbasoffd.end(),   data2.end()));

    thrust::transform(first, last, data.begin(), zipup());

    thrust::transform(data.begin()+1, data.end(), data.begin(), dout.begin(), thrust::minus<Numeric>());
    double_moving_average(m,n,dout, w3, data2);

    first0 = thrust::make_zip_iterator(thrust::make_tuple(data.begin(), data2.begin()));
    first1 = thrust::make_zip_iterator(thrust::make_tuple(data.begin() + 1, data2.begin() + 1));
    last0  = thrust::make_zip_iterator(thrust::make_tuple(data.end(),  data2.end()));
    
    thrust::transform(first0, last0, first1, dout.begin(), compare_zip());
    segmented_peak_sums(m, n, avgbasoffp, pintegralp);
    segmented_peak_sums(m, n, avgbasoffd, pintegrald);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// Test of peak mask
////////////////////////////////////////////////////////////////////////////////////////////////////
int main(void)
{

  int values[NDATA] = {	7, 1, 16, 18, 19, 20, 32, 53, 12, 11, 10, 10, 9, 5, 4, 1, 1, 3, 6, 1};
  int m = 10, n = 2, w1 = 3, w2 = 5, w3 = 2;
   
  // transfer to device
  thrust::device_vector<Numeric> idata(values, values + NDATA);
  cout << "testing peak mask and integral" << endl;
  cout << "idata (two vectors one after another):" << endl;
  printvec(m,n,idata);

  thrust::device_vector<Numeric> orgbasoffp(m*n);
  thrust::device_vector<Numeric> orgbasoffd(m*n);
  thrust::device_vector<Numeric> avgbasoffp(m*n);
  thrust::device_vector<Numeric> avgbasoffd(m*n);
  thrust::device_vector<Numeric> pintegralp(m*n);
  thrust::device_vector<Numeric> pintegrald(m*n);
  thrust::device_vector<Numeric> pout(m*n);
  thrust::device_vector<Numeric> dout(m*n);
  
  thrust::device_vector<Numeric> data(m*n);
  thrust::device_vector<Numeric> data2(m*n);

  double_moving_average(m,n,idata, w2, data2);
  
  //original peaks up and down
  create_peaks(idata, data2, orgbasoffp, orgbasoffd);

  cout << "data in peak up:" << endl;
  printvec(1,NDATA,orgbasoffp); 
  cout << "data in peak down:" << endl;
  printvec(1,NDATA,orgbasoffd); 
  
  double_moving_average(m,n,idata, w1, data);
  
  //average peaks up and down
  create_peaks(data, data2, avgbasoffp, avgbasoffd);
  
  cout << "avg data in peak up:" << endl;
  printvec(1,NDATA,avgbasoffp); 
  cout << "avg data in peak down:" << endl;
  printvec(1,NDATA,avgbasoffd); 
  
  segmented_peak_sums(m, n, avgbasoffp, pintegralp);
  cout << "integral in peak up:" << endl;
  printvec(1,NDATA,pintegralp); 
  
  segmented_peak_sums(m, n, avgbasoffd, pintegrald);
  cout << "integral in peak down:" << endl;
  printvec(1,NDATA,pintegrald); 
  
  //orgbasoffp, avgbasoffp
  avg_find_peaks(m, n, w1, w2, w3, orgbasoffp, avgbasoffp, pout);

  cout << "pout in peak up:" << endl;
  printvec(1,NDATA,pout); 
 
  //orgbasoffd, avgbasoffd
  avg_find_peaks(m, n, w1, w2, w3, orgbasoffd, avgbasoffd, dout);

  cout << "dout in peak down:" << endl;
  printvec(1,NDATA,dout); 
  
  
  find_doublemaxpeaks_opt( m, n, idata, w1, w2, w3, pout, avgbasoffp, pintegralp, dout, avgbasoffd, pintegrald);
  
  cout << "avg.orig data in peak up:" << endl;
  printvec(1,NDATA,avgbasoffp); 
  cout << "avg.orig data in peak down:" << endl;
  printvec(1,NDATA,avgbasoffd); 
  cout << "integral.orig in peak up:" << endl;
  printvec(1,NDATA,pintegralp); 
  cout << "integral.orig in peak down:" << endl;
  printvec(1,NDATA,pintegrald); 
  cout << "pout.orig in peak up:" << endl;
  printvec(1,NDATA,pout); 
  cout << "dout.orig in peak down:" << endl;
  printvec(1,NDATA,dout); 
  
 
  return 0;
}

