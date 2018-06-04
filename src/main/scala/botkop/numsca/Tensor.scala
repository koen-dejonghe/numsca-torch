package botkop.numsca

import jtorch.cpu.NDArray

case class Tensor(array: NDArray){

}

object Tensor {

  def apply(data: Array[Float]): Tensor = {
    Tensor(NDArray.create(data))
  }

}
