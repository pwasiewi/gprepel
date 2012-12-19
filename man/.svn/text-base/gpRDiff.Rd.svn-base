\name{gpRDiff}
\alias{gpRDiff}
\title{Compute difference between shifted columns by w in a matrix with GPU}

\description{
    Compute difference between shifted columns by w in a column-major matrix with GPU
}

\usage{
    gpRDiff(points,w)
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
    gpRPostmave, gpRPremave, gpRMoverage, gpRBasavoff
}

\examples{
numVectors <- 10
dimension <- 5
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
gpRDiff(Vectors,2,5)
}

\keyword{math}
\keyword{array & algebra}