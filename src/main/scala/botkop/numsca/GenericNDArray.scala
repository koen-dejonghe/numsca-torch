package botkop.numsca

import scala.language.experimental.macros
import torch.cpu.{TH, THByteTensor, THDoubleTensor, THFloatTensor}

trait GenericNDArray {
  def nDimension: Int
}

class FloatNDArray(payload: THFloatTensor) extends GenericNDArray {
  override def nDimension: Int = TH.THFloatTensor_nDimension(payload)
}

class DoubleNDArray(payload: THDoubleTensor) extends GenericNDArray {
  override def nDimension: Int = TH.THDoubleTensor_nDimension(payload)
}

class ByteNDArray(payload: THByteTensor) extends GenericNDArray {
  override def nDimension: Int = TH.THByteTensor_nDimension(payload)
}

object GenericNDArray {

}
