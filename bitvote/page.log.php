   <main>
      <nav>
<?php
$sql='SELECT count(id) AS t FROM issue_votes';
$sql=SQL2ARRAY($sql)[0][0];
for ($i=floor($sql/10000); $i>=0; $i--) { ?>
         <li><a href="?page=log&p=<?=$i?>"><?=SQLATE('page', $LANG, true)?> <?=$i?></a></li>
<?php } ?>
      </nav>
      <section>
<?php
$sql='SELECT i.id, i.ID_issue, v.';
if ($LANG=='fr')
   $sql.='fr';
$sql.='caption as caption, i.stamp, v.frcaption, i.ID_user, CONCAT(u.ssid, u.mpid) AS mky FROM issue_votes AS i LEFT JOIN votes AS v ON v.id=i.ID_vote LEFT JOIN users AS u ON u.id=i.ID_user';
if (isset($_REQUEST['search']) && $_REQUEST['search']!='') {
   $search=str_replace('%', '', $search);
   $search=SQLIFY($_REQUEST['search']);
   $sql.=' LEFT JOIN issues AS si ON si.id=i.ID_issue';
   $sql.=' LEFT JOIN regions AS r ON r.id=u.ID_region';
   $sql.=' LEFT JOIN parties AS p ON p.id=u.ID_party';
   $sql.=' WHERE (';
   $sql.='v.caption LIKE "%'.$search.'%"';
   $sql.=' OR v.frcaption LIKE "%'.$search.'%"';
   $sql.=' OR si.id LIKE "%'.$search.'%"';
   $sql.=' OR si.title LIKE "%'.$search.'%"';
   $sql.=' OR si.frtitle LIKE "%'.$search.'%"';
   $sql.=' OR si.journal LIKE "%'.$search.'%"';
   $sql.=' OR si.leginfo LIKE "%'.$search.'%"';
   $sql.=' OR si.ssid LIKE "%'.$search.'%"';
   $sql.=' OR si.mpid LIKE "%'.$search.'%"';
   $sql.=' OR r.caption LIKE "%'.$search.'%"';
   $sql.=' OR r.frcaption LIKE "%'.$search.'%"';
   $sql.=' OR p.caption LIKE "%'.$search.'%"';
   $sql.=' OR p.frcaption LIKE "%'.$search.'%"';
   $sql.=') ORDER BY i.id LIMIT 10000';
//   echo "<h5>".$sql."</h5>";
} else {
   $sql.=' ORDER BY i.id LIMIT 10000';
}
if (isset($_REQUEST['p'])) 
   $sql.=' OFFSET '.intval($_REQUEST['p'])*10000;
//die($sql);
$sql=SQL2ARRAY($sql);
if (isset($_REQUEST['search']) && $_REQUEST['search']!='') {
   echo '<h2>'.SQLATE('filtered', $LANG, true).'! - '.count($sql).' '.SQLATE('results', $LANG, false).'</h2>';
   if (count($sql)==0)
      echo '<h2>('.SQLATE('Unable to search by hashes', $LANG, false).')</h2>';
}
?>
         <table>
            <tr>
               <th><?=SQLATE('vote', $LANG, true)?></th>
               <th><?=SQLATE('motion', $LANG, true)?></th>
               <th><?=SQLATE('hash', $LANG, true)?></th>
<?php foreach ($sql as $s) { ?>
            </tr><tr>
               <td>#<?=$s['id']?>: <?=$s['caption']?></td>
               <td><a href="?page=issue&id=<?=$s['ID_issue']?>"><?=$s['ID_issue']?></a></td>
               <td><?=substr(hash('sha512',$salt[0].$s['id'].$s['stamp'].substr('.'.$sql['mky'],-10,4).$s['frcaption'].$s['ID_user'].$salt[1]),-50)?></td>
<?php } // + $sql['stamp'] ?>
         </tr></table>
      </section>
   </main>
