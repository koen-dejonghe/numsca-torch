package ns

import com.typesafe.scalalogging.LazyLogging
import torch.cpu._

import scala.language.{implicitConversions, postfixOps}

class Tensor private[ns](val array: THFloatTensor) extends LazyLogging {

  def dim: Int = array.getNDimension

  def shape: List[Int] = {
    val s = CInt64Array.frompointer(array.getSize)
    (0 until dim).toList.map(i => s.getitem(i).toInt)
  }

  def stride: List[Int] = {
    val s = CInt64Array.frompointer(array.getStride)
    (0 until dim).toList.map(i => s.getitem(i).toInt)
  }

  // val size: Int = shape.product
  def size: Int = numel
  def realSize: Int =
    shape.zip(stride).map { case (d, s) => if (s == 0) 1 else d }.product

  def desc: String = TH.THFloatTensor_desc(array).getStr
  def numel: Int = TH.THFloatTensor_numel(array)

  override def toString: String =
    s"tensor of shape $shape and stride $stride ($realSize / $size)\n" + data.toList

  def data: Array[Float] = ns.data(this)

  def value(ix: List[Int]): Float = ns.getValue(this, ix)
  def value(ix: Int*): Float = ns.getValue(this, ix.toList)

  override def finalize(): Unit = {
//    val memSize = MemoryManager.dec(size) // obtain size before freeing!
//    logger.debug(s"freeing (mem = $memSize)")
    logger.debug(s"finalizing")
    array.delete()
  }

  def copy(): Tensor = ns.copy(this)

  def reshape(newShape: List[Int]): Tensor =
    ns.reshape(this, newShape)

  def reshape(newShape: Int*): Tensor = reshape(newShape.toList)

  def squeeze(): Tensor = ns.squeeze(this)

  def apply(i: Int*): Tensor = ns.narrow(this, i.toList)
  def apply(rs: NumscaRange*)(implicit z: Int = 0): Tensor =
    ns.narrow(this, NumscaRangeSeq(rs))
  def select(ixs: Tensor*): Tensor = ns.indexSelect(this, ixs.toSeq)

  def isSameAs(a: Tensor): Boolean = ns.equal(this, a)

  def :=(f: Number): Unit = ns.assign(this, f)
  def :=(a: Tensor): Unit = ns.assign(this, a)

  def +=(f: Number): Unit = ns.addi(this, f)
  def +=(a: Tensor): Unit = ns.addi(this, a)

  def -=(f: Number): Unit = ns.subi(this, f)
  def -=(a: Tensor): Unit = ns.subi(this, a)

  def *=(f: Number): Unit = ns.muli(this, f)
  def *=(a: Tensor): Unit = ns.muli(this, a)

  def /=(f: Number): Unit = ns.divi(this, f)
  def /=(a: Tensor): Unit = ns.divi(this, a)

  def **=(f: Number): Unit = ns.powi(this, f)
  def **=(a: Tensor): Unit = ns.powi(this, a)

  def +(f: Number): Tensor = ns.add(this, f)
  def +(a: Tensor): Tensor = ns.add(this, a)

  def -(f: Number): Tensor = ns.sub(this, f)
  def -(a: Tensor): Tensor = ns.sub(this, a)

  def *(f: Number): Tensor = ns.mul(this, f)
  def *(a: Tensor): Tensor = ns.mul(this, a)

  def /(f: Number): Tensor = ns.div(this, f)
  def /(a: Tensor): Tensor = ns.div(this, a)

  def **(f: Number): Tensor = ns.pow(this, f)
  def **(a: Tensor): Tensor = ns.pow(this, a)

  def unary_- : Tensor = ns.neg(this)
}

object Tensor extends LazyLogging {

  def apply(data: Array[Float]): Tensor = ns.create(data)
  def apply(data: Number*): Tensor = Tensor(data.map(_.floatValue()).toArray)

  implicit def toRawTensor(t: Tensor): THFloatTensor = t.array

}
