package botkop.numsca

import java.util.concurrent.atomic.AtomicLong

import com.typesafe.scalalogging.LazyLogging

object MemoryManager extends LazyLogging {

  val Threshold: Long = 2L * 1024L * 1024L * 1024L // 2 GB
  val FloatSize = 4
  private val hiMemMark = new AtomicLong(0)

  def dec(size: Long): Long = hiMemMark.addAndGet(-size * FloatSize)
  def inc(size: Long): Long = hiMemMark.addAndGet(size * FloatSize)

  def memCheck(size: Long): Unit = {
    val level = inc(size)
    if (level > Threshold) {
      logger.debug(s"invoking gc ($level/$Threshold)")
      System.gc()
    }
  }

  def memCheck(shape: List[Int]): Unit = memCheck(shape.product)
}
