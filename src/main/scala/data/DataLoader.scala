package data

import ns.{LongTensor, Tensor}

trait DataLoader extends scala.collection.immutable.Iterable[(Tensor, Tensor)] {
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

