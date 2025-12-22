<?php
   $sql='SELECT i.id, i.deadline, i.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='title as title';
   if (isset($_SESSION['UID'])) {
      $sql.=', v.';
      if ($LANG=='fr')
         $sql.='fr';
      $sql.='caption as caption';
   }
   $sql.=' FROM issues AS i ';
   if (isset($_SESSION['UID']))
      $sql.='LEFT JOIN issue_votes AS iv ON iv.ID_user="'
         .$_SESSION['UID'].'" AND iv.ID_issue=i.id'
         .' LEFT JOIN votes AS v ON v.id=iv.ID_vote';
   if (isset($_REQUEST['search']) && $_REQUEST['search']!='') {
      $search=str_replace('%', '', $_REQUEST['search']);
      $search=SQLIFY($_REQUEST['search']);
      $sql.=' WHERE (';
      $sql.='i.title LIKE "%'.$search.'%"';
      $sql.=' OR i.title LIKE "%'.$search.'%"';
      $sql.=' OR i.frtitle LIKE "%'.$search.'%"';
      $sql.=' OR i.journal LIKE "%'.$search.'%"';
      $sql.=' OR i.leginfo LIKE "%'.$search.'%"';
      $sql.=' OR i.ssid LIKE "%'.$search.'%"';
      $sql.=' OR i.mpid LIKE "%'.$search.'%"';
      $sql.=' OR i.id LIKE "%'.$search.'%"';
      //$sql.=' OR v.caption LIKE "%'.$search.'%"';
      //$sql.=' OR v.frcaption LIKE "%'.$search.'%"';
      $sql.=')';
   }
   $sql.=' ORDER BY deadline DESC';
   $sql=SQL2ARRAY($sql);
?>
   <main>
      <h3><?=SQLATE('motions', $LANG, true)?></h3>
      <section>
         <table id="tblVotes">
               <tr>
                  <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 0)">Date</th>
                  <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 1)">Title</th>
<?php if (isset($_SESSION['UID'])) {?>
                  <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 2)">Your vote</th>
<?php } ?>
               </tr>
<?php foreach ($sql as $s) { ?>
               <tr>
                  <td>
<span><?=$s['deadline']?></span>
                  </td>
                  <td>
<a href='?page=issue&id=<?=$s['id']?>'><?=$s['title']?></a>
                  </td>
<?php if (isset($_SESSION['UID'])) {?>
                  <td>
<?=$s['caption']?>
                  </td>
<?php } ?>
               </tr>
<?php } ?>
         </table>
      </section>
   </main>
