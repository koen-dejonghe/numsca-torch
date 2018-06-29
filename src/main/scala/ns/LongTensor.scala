package ns

import com.typesafe.scalalogging.LazyLogging
import torch.cpu.{THJNI, THLongTensor}

class LongTensor(val array: THLongTensor) extends LazyLogging {
  var pointer: Long = LongTensor.pointer(array)
  override def finalize(): Unit = {
    logger.debug(s"freeing long tensor $pointer")
    THJNI.THLongTensor_free(pointer, array)
  }
}

object LongTensor extends THLongTensor {
  def pointer(t: THLongTensor): Long = THLongTensor.getCPtr(t)

  def apply(ft: Tensor): LongTensor =
    new LongTensor(ns.floatTensorToLongTensor(ft))
}
