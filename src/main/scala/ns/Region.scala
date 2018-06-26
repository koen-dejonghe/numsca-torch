package ns

import com.typesafe.scalalogging.LazyLogging

import scala.collection.mutable.ArrayBuffer

class Region private () extends LazyLogging {
  private val registered = ArrayBuffer.empty[() => Unit]
  def register(finalizer: () => Unit): Unit = registered += finalizer
  def releaseAll(): Unit = {
    registered.foreach(f => f()) // todo - will leak if f() throws
    logger.debug(s"Released ${registered.size} tensors. Remaining: ${MemoryManager.hiTensorMark.get}.")
  }
}

object Region {
  def run[A](f: Region => A): A = {
    val r = new Region
    try f(r) finally r.releaseAll()
  }
}

