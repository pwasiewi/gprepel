\name{gpRBasavoff}
\alias{gpRBasavoff}
\title{Compute moving average minus baseline with GPU}

\description{
    Computes double moving average (in windows [i-1:i-w1] [i+1:i+w1] and divides their sum by 2) minus baseline (double moving average with windows width w2) with GPU
}

\usage{
    gpRBasavoff(points,w)
}

\arguments{
    \item{points}{a matrix of floating point numbers in which each column is a vector.}
	\item{w}{window width of double moving average w1}
	\item{w}{window width of double moving average w2 (typically w2 >> w1)}
}

\details{
    Compute moving average in columns
}

\value{
    a matrix of floating point numbers in which each column is a vector
}

\seealso{
    gpRPostmave,gpRPremave
}

\examples{
numVectors <- 10
dimension <- 5
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
gpRBasavoff(Vectors,2,5)
}

\keyword{math}
\keyword{array & algebra}



