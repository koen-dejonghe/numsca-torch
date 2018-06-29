package ns

import com.typesafe.scalalogging.LazyLogging
import torch.cpu.{THByteTensor, THJNI}

class ByteTensor(val array: THByteTensor) extends LazyLogging {
  var pointer: Long = ByteTensor.pointer(array)

  override def finalize(): Unit = {
    logger.debug(s"freeing byte tensor $pointer")
    THJNI.THByteTensor_free(pointer, array)
  }
}

object ByteTensor extends THByteTensor {
  def pointer(t: THByteTensor): Long = THByteTensor.getCPtr(t)

  def apply(ft: Tensor): ByteTensor = {
    new ByteTensor(ns.floatTensorToByteTensor(ft))
  }
}

