package data

import ns.Tensor

trait DataLoader extends Iterable[(Tensor, Tensor)] {
  def numSamples: Int
  def numBatches: Int
}

object DataLoader {
  def instance(dataSet: String,
               mode: String,
               miniBatchSize: Int,
               take: Option[Int] = None): DataLoader =
    dataSet match {
      case "mnist" =>
        new MnistDataLoader(mode, miniBatchSize, take)
    }
}

